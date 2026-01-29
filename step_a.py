import cv2
import numpy as np
import os
import glob

SCENE_FOLDER = "images/scenes"
MODEL_FOLDER = "images/models"
OUTPUT_FOLDER = "output_step_a"


def solve_step_a():

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # We only process the 'easy' scenes for Step A (e1...e5)
    scene_paths = sorted(glob.glob(os.path.join(SCENE_FOLDER, "e*.png")))
    model_paths = sorted(glob.glob(os.path.join(MODEL_FOLDER, "*.jpg")))

    # Initialize SIFT and FLANN
    sift = cv2.SIFT_create()
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for scene_path in scene_paths:
        scene_name = os.path.basename(scene_path)
        print(f"\nProcessing Scene: {scene_name}")

        # Load grayscale for processing, Color for visualization
        scene_gray = cv2.imread(scene_path, 0)
        scene_rgb = cv2.imread(scene_path)

        if scene_gray is None:
            continue

        kp_scene, des_scene = sift.detectAndCompute(scene_gray, None)

        for model_path in model_paths:
            product_name = os.path.basename(model_path)
            model_img = cv2.imread(model_path, 0)
            if model_img is None:
                continue

            kp_model, des_model = sift.detectAndCompute(model_img, None)

            if (
                des_model is None
                or des_scene is None
                or len(des_model) < 2
                or len(des_scene) < 2
            ):
                continue

            matches = flann.knnMatch(des_model, des_scene, k=2)

            # Lowe's Ratio Test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            # Homography Check
            MIN_MATCH_COUNT = 10
            if len(good_matches) <= MIN_MATCH_COUNT:
                continue

            src_pts = np.float32([kp_model[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is None:
                continue

            h, w = model_img.shape
            # Define corners of the object
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

            # Project corners to scene
            dst = cv2.perspectiveTransform(pts, M)

            # 1. VISUALIZATION: Draw Green Box
            # cv2.polylines handles rotation/perspective skew correctly
            scene_rgb = cv2.polylines(
                scene_rgb, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA
            )

            # 2. CONSOLE OUTPUT
            x, y, w_box, h_box = cv2.boundingRect(dst)
            center_x = int(x + w_box / 2)
            center_y = int(y + h_box / 2)
            print(f"Product {product_name} - 1 instance found:")
            print(f"\tInstance 1 {{position: ({center_x},{center_y}), width: {w_box}px, height: {h_box}px}}")

        # Save the annotated image
        output_path = os.path.join(OUTPUT_FOLDER, scene_name)
        cv2.imwrite(output_path, scene_rgb)
        print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    solve_step_a()
