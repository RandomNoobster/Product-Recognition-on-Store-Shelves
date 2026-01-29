import cv2
import numpy as np
import os
import glob

# --- SHARED HELPERS ---

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def list_scene_paths(scene_folder, pattern):
    return sorted(glob.glob(os.path.join(scene_folder, pattern)))


def list_model_paths(model_folder, allowed_names=None):
    model_paths = sorted(glob.glob(os.path.join(model_folder, '*.jpg')))
    if not allowed_names:
        return model_paths

    allowed_set = set()
    for name in allowed_names:
        if isinstance(name, int):
            allowed_set.add(f"{name}.jpg")
        else:
            allowed_set.add(name if name.endswith('.jpg') else f"{name}.jpg")

    return [path for path in model_paths if os.path.basename(path) in allowed_set]


def create_sift():
    return cv2.SIFT_create(nfeatures=0, contrastThreshold=0.01, edgeThreshold=15)


def create_flann():
    return cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))


def compute_ghtt_votes(kp_model, kp_scene, matches, model_center):
    """Star Model Voting (GHT) - Slide 28 of Instance Detection"""
    votes = []
    vote_matches = []
    for m in matches:
        k_m = kp_model[m.queryIdx]
        k_s = kp_scene[m.trainIdx]
        scale = k_s.size / k_m.size
        theta = np.deg2rad(k_s.angle - k_m.angle)
        v_x = model_center[0] - k_m.pt[0]
        v_y = model_center[1] - k_m.pt[1]
        rot_v_x = v_x * np.cos(theta) - v_y * np.sin(theta)
        rot_v_y = v_x * np.sin(theta) + v_y * np.cos(theta)
        guess_x = k_s.pt[0] + rot_v_x * scale
        guess_y = k_s.pt[1] + rot_v_y * scale
        votes.append([guess_x, guess_y])
        vote_matches.append(m)
    return np.array(votes), vote_matches


def cluster_votes_greedy(votes, match_list, distance_thresh=30, min_votes=5):
    """Greedy clustering to group votes for object centers"""
    clusters = []
    used = np.zeros(len(votes), dtype=bool)
    for i in range(len(votes)):
        if used[i]:
            continue
        current_cluster_indices = [i]
        used[i] = True
        seed_point = votes[i]
        dists = np.linalg.norm(votes - seed_point, axis=1)
        neighbors = np.where((dists < distance_thresh) & (~used))[0]
        current_cluster_indices.extend(neighbors.tolist())
        used[neighbors] = True
        if len(current_cluster_indices) >= min_votes:
            clusters.append(current_cluster_indices)
    return clusters


def get_color_similarity(model_bgr, scene_bgr, dst_pts):
    """HSV histogram correlation to distinguish variants."""
    try:
        h_m, w_m = model_bgr.shape[:2]
        pts_std = np.float32([[0, 0], [0, h_m - 1], [w_m - 1, h_m - 1], [w_m - 1, 0]])

        M = cv2.getPerspectiveTransform(dst_pts, pts_std)
        scene_patch = cv2.warpPerspective(scene_bgr, M, (w_m, h_m))

        model_hsv = cv2.cvtColor(model_bgr, cv2.COLOR_BGR2HSV)
        scene_hsv = cv2.cvtColor(scene_patch, cv2.COLOR_BGR2HSV)

        hist_bins = [8, 4]
        ranges = [0, 180, 0, 256]

        hist_model = cv2.calcHist([model_hsv], [0, 1], None, hist_bins, ranges)
        hist_scene = cv2.calcHist([scene_hsv], [0, 1], None, hist_bins, ranges)

        cv2.normalize(hist_model, hist_model, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_scene, hist_scene, 0, 1, cv2.NORM_MINMAX)

        similarity = cv2.compareHist(hist_model, hist_scene, cv2.HISTCMP_CORREL)
        return max(0.0, similarity)
    except Exception:
        return 0.0


def is_geometrically_valid(dst_pts, scene_h, scene_w):
    """Checks if the box is plausible (roughly rectangular and convex)."""
    if not cv2.isContourConvex(np.int32(dst_pts)):
        return False
    x, y, w, h = cv2.boundingRect(dst_pts)
    if w < 30 or h < 30:
        return False

    pts = dst_pts.reshape(4, 2)
    dists = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    if max(dists[0], dists[2]) / min(dists[0], dists[2]) > 1.25:
        return False
    if max(dists[1], dists[3]) / min(dists[1], dists[3]) > 1.25:
        return False

    return True


def nms_boxes_robust(candidates, scene_shape):
    """Global NMS across all product models."""
    if not candidates:
        return []
    candidates.sort(key=lambda x: x['final_score'], reverse=True)

    keep = []
    while len(candidates) > 0:
        best = candidates.pop(0)
        keep.append(best)
        remaining = []
        for other in candidates:
            dist = np.linalg.norm(np.array(best['pos']) - np.array(other['pos']))
            if dist < 60:
                continue

            mask1 = np.zeros(scene_shape, dtype=np.uint8)
            mask2 = np.zeros(scene_shape, dtype=np.uint8)
            cv2.fillPoly(mask1, [np.int32(best['dst'])], 1)
            cv2.fillPoly(mask2, [np.int32(other['dst'])], 1)
            inter = np.sum(np.logical_and(mask1, mask2))
            union = np.sum(np.logical_or(mask1, mask2))
            if inter / union > 0.1:
                continue

            remaining.append(other)
        candidates = remaining
    return keep


def detect_products_in_scene(scene_path, model_paths, sift, flann, verbose=False):
    scene_gray = cv2.imread(scene_path, 0)
    scene_rgb = cv2.imread(scene_path)
    if scene_gray is None or scene_rgb is None:
        return None, []

    scene_gray = cv2.bilateralFilter(scene_gray, d=9, sigmaColor=75, sigmaSpace=75)
    scene_h, scene_w = scene_gray.shape

    kp_scene, des_scene = sift.detectAndCompute(scene_gray, None)
    all_candidates = []

    for model_path in model_paths:
        product_name = os.path.basename(model_path)
        model_bgr = cv2.imread(model_path)
        if model_bgr is None:
            continue
        model_gray = cv2.cvtColor(model_bgr, cv2.COLOR_BGR2GRAY)
        model_gray = cv2.bilateralFilter(model_gray, d=9, sigmaColor=75, sigmaSpace=75)
        h_m, w_m = model_gray.shape
        model_center = (w_m / 2, h_m / 2)

        kp_model, des_model = sift.detectAndCompute(model_gray, None)
        if des_model is None or des_scene is None:
            continue
        matches = flann.knnMatch(des_model, des_scene, k=2)

        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 5:
            continue

        votes, m_list = compute_ghtt_votes(kp_model, kp_scene, good, model_center)
        clusters = cluster_votes_greedy(votes, m_list, distance_thresh=40, min_votes=5)

        for idxs in clusters:
            c_matches = [m_list[i] for i in idxs]
            src = np.float32([kp_model[m.queryIdx].pt for m in c_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in c_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                continue

            inl = mask.ravel() == 1
            if np.sum(inl) >= 4:
                M_ref, _ = cv2.findHomography(src[inl], dst_pts[inl], 0)
                if M_ref is not None:
                    M = M_ref

            corners = np.float32([[0, 0], [0, h_m - 1], [w_m - 1, h_m - 1], [w_m - 1, 0]]).reshape(-1, 1, 2)
            try:
                proj = cv2.perspectiveTransform(corners, M)
                if is_geometrically_valid(proj, scene_h, scene_w):
                    _, _, w_b, h_b = cv2.boundingRect(proj)

                    color_sim = get_color_similarity(model_bgr, scene_rgb, proj)
                    area = w_b * h_b
                    final_score = np.sum(mask) * (color_sim) * np.sqrt(area)

                    if verbose:
                        print(
                            f"  Detected {product_name} with score {final_score:.2f} "
                            f"(Inliers: {np.sum(mask)}, ColorSim: {color_sim:.2f}, Area: {area})"
                        )

                    all_candidates.append({
                        'dst': proj,
                        'final_score': final_score,
                        'name': product_name,
                        'pos': (int(np.mean(proj[:, 0, 0])), int(np.mean(proj[:, 0, 1]))),
                        'w': w_b,
                        'h': h_b
                    })
            except Exception:
                continue

    final = nms_boxes_robust(all_candidates, (scene_h, scene_w))
    return scene_rgb, final


def draw_detections(scene_rgb, detections, box_color=(0, 255, 0), label_color=(0, 0, 255),
                    box_thickness=3, font_scale=0.7, font_thickness=2, line_type=8):
    for det in detections:
        cv2.polylines(scene_rgb, [np.int32(det['dst'])], True, box_color, box_thickness, line_type)
        cv2.putText(scene_rgb, det['name'][:-4], det['pos'], cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, label_color, font_thickness)


def group_detections_by_product(detections):
    grouped = {}
    for det in detections:
        key = os.path.splitext(det['name'])[0]
        grouped.setdefault(key, []).append(det)
    return grouped
