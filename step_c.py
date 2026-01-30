import cv2
import os

from utils import (
    ensure_dir,
    list_scene_paths,
    list_model_paths,
    create_sift,
    create_flann,
    detect_products_in_scene,
    draw_detections,
    group_detections_by_product,
)

SCENE_FOLDER = "images/scenes"
MODEL_FOLDER = "images/models"
OUTPUT_FOLDER = "output_step_c"

def solve_step_c():
    ensure_dir(OUTPUT_FOLDER)
    
    scene_paths = list_scene_paths(SCENE_FOLDER, "h*.jpg")
    # allowed_models = [f"{i}.jpg" for i in range(24)]
    allowed_models = None
    model_paths = list_model_paths(MODEL_FOLDER, allowed_models)

    sift = create_sift()
    flann = create_flann()

    for scene_path in scene_paths:
        scene_name = os.path.basename(scene_path)
        print(f"\nProcessing Scene: {scene_name}")

        scene_rgb, detections = detect_products_in_scene(
            scene_path,
            model_paths,
            sift,
            flann,
            verbose=True
        )
        if scene_rgb is None:
            continue

        grouped = group_detections_by_product(detections)
        print(f"  Found {len(detections)} unique objects.")
        for product_name, dets in grouped.items():
            print(f"Product {product_name} - {len(dets)} instance found:")
            for i, det in enumerate(dets, start=1):
                print(
                    f"\tInstance {i} {{position: {det['pos']}, "
                    f"width: {det['w']}px, height: {det['h']}px}}"
                )

        draw_detections(scene_rgb, detections, box_thickness=2, font_scale=0.5, font_thickness=1)

        output_path = os.path.join(OUTPUT_FOLDER, scene_name)
        cv2.imwrite(output_path, scene_rgb)
        print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    solve_step_c()
