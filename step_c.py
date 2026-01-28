import cv2
import numpy as np
import os
import glob

# --- HELPER FUNCTIONS ---

def compute_ghtt_votes(kp_model, kp_scene, matches, model_center):
    """ Star Model Voting (GHT) """
    votes = []
    vote_matches = [] 
    for m in matches:
        k_m = kp_model[m.queryIdx]
        k_s = kp_scene[m.trainIdx]

        scale = k_s.size / k_m.size
        # SIFT angle is in degrees
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

def cluster_votes_greedy(votes, match_list, distance_thresh=40, min_votes=3):
    """
    Greedy Clustering (Distance based).
    Thresholds lowered for Step C (Hard)
    """
    clusters = [] 
    used = np.zeros(len(votes), dtype=bool)

    for i in range(len(votes)):
        if used[i]: continue
        
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

def calculate_iou(pts1, pts2, shape):
    """ Intersection over Union for NMS """
    mask1 = np.zeros(shape, dtype=np.uint8)
    mask2 = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask1, [np.int32(pts1)], 1)
    cv2.fillPoly(mask2, [np.int32(pts2)], 1)
    
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    area_inter = np.sum(intersection)
    area_union = np.sum(union)
    
    return area_inter / area_union if area_union > 0 else 0

def is_plausible_box(dst_pts, min_side=10, max_side=1000):
    """ Geometric sanity check for Step C (allowed to be smaller/larger) """
    if not cv2.isContourConvex(np.int32(dst_pts)): return False
    x, y, w, h = cv2.boundingRect(dst_pts)
    if w < min_side or h < min_side: return False
    if w > max_side or h > max_side: return False
    return True

# --- MAIN LOGIC ---

def solve_step_c():
    SCENE_FOLDER = 'images/scenes'
    MODEL_FOLDER = 'images/models'
    OUTPUT_FOLDER = 'output_step_c'
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # Load Hard Scenes
    scene_paths = sorted(glob.glob(os.path.join(SCENE_FOLDER, 'h*.jpg')))
    model_paths = sorted(glob.glob(os.path.join(MODEL_FOLDER, '*.jpg')))

    print(f"Found {len(scene_paths)} scenes and {len(model_paths)} models.")

    sift = cv2.SIFT_create()
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for scene_path in scene_paths:
        scene_name = os.path.basename(scene_path)
        print(f"\nProcessing Scene: {scene_name}...")
        
        scene_gray = cv2.imread(scene_path, 0)
        scene_rgb = cv2.imread(scene_path)
        if scene_gray is None: continue
        
        scene_h, scene_w = scene_gray.shape
        kp_scene, des_scene = sift.detectAndCompute(scene_gray, None)
        
        total_detections_in_scene = 0

        for model_path in model_paths:
            product_name = os.path.basename(model_path)
            # Optional: Uncomment to see detailed progress
            # print(f"  Checking {product_name}...", end='\r') 
            
            model_img = cv2.imread(model_path, 0)
            if model_img is None: continue
            
            h_m, w_m = model_img.shape
            model_center = (w_m / 2, h_m / 2)

            kp_model, des_model = sift.detectAndCompute(model_img, None)
            if des_model is None or des_scene is None: continue

            # 1. Matching
            matches = flann.knnMatch(des_model, des_scene, k=2)

            good_matches = []
            for m, n in matches:
                # 0.8 Ratio Test (More permissive for Hard Step C)
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)

            # Lowered requirement: Need only 4 matches to attempt detection
            if len(good_matches) < 4: continue

            # 2. GHT Voting
            votes, match_list = compute_ghtt_votes(kp_model, kp_scene, good_matches, model_center)
            
            # 3. Greedy Clustering (dist=40, min_votes=4)
            cluster_indices_list = cluster_votes_greedy(votes, match_list, distance_thresh=40, min_votes=4)

            candidates = []

            for indices in cluster_indices_list:
                cluster_matches = [match_list[i] for i in indices]

                src_pts = np.float32([kp_model[m.queryIdx].pt for m in cluster_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in cluster_matches]).reshape(-1, 1, 2)

                # 4. RANSAC (Threshold=10.0 for "loose" fit in blurry scenes)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

                if M is not None:
                    # 5. Refinement (Least Squares on Inliers)
                    inliers_mask = mask.ravel() == 1
                    if np.sum(inliers_mask) >= 4:
                        src_in = src_pts[inliers_mask]
                        dst_in = dst_pts[inliers_mask]
                        M_refined, _ = cv2.findHomography(src_in, dst_in, 0)
                        if M_refined is not None: M = M_refined

                    pts = np.float32([[0, 0], [0, h_m-1], [w_m-1, h_m-1], [w_m-1, 0]]).reshape(-1, 1, 2)
                    try:
                        dst = cv2.perspectiveTransform(pts, M)
                        
                        # 6. Plausibility Check
                        if is_plausible_box(dst, min_side=15, max_side=max(scene_h, scene_w)):
                            x, y, w_box, h_box = cv2.boundingRect(dst)
                            center = (int(x + w_box/2), int(y + h_box/2))
                            score = np.sum(inliers_mask)
                            
                            candidates.append({
                                'dst': dst,
                                'score': score,
                                'pos': center,
                                'w': w_box,
                                'h': h_box
                            })
                    except: pass

            # 7. NMS (Remove overlaps)
            candidates.sort(key=lambda x: x['score'], reverse=True)
            unique_instances = []
            
            for cand in candidates:
                is_duplicate = False
                for existing in unique_instances:
                    # Center distance check (20px)
                    dist = np.linalg.norm(np.array(cand['pos']) - np.array(existing['pos']))
                    if dist < 20: 
                        is_duplicate = True
                        break
                    
                    # IoU check
                    iou = calculate_iou(cand['dst'], existing['dst'], (scene_h, scene_w))
                    if iou > 0.15:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_instances.append(cand)

            # 8. Output & Draw
            if len(unique_instances) > 0:
                total_detections_in_scene += len(unique_instances)
                print(f"  > Found {len(unique_instances)} instance(s) of {product_name}")
                for i, inst in enumerate(unique_instances):
                    # Draw Green Box
                    scene_rgb = cv2.polylines(scene_rgb, [np.int32(inst['dst'])], True, (0, 255, 0), 2, cv2.LINE_AA)
                    # Label
                    cv2.putText(scene_rgb, f"{product_name[:-4]}", (inst['pos'][0], inst['pos'][1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        print(f"  Total detections in scene: {total_detections_in_scene}")
        output_path = os.path.join(OUTPUT_FOLDER, scene_name)
        cv2.imwrite(output_path, scene_rgb)
        print(f"  Saved visualization to {output_path}")

if __name__ == "__main__":
    solve_step_c()