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
)

SCENE_FOLDER = "images/scenes"
MODEL_FOLDER = "images/models"
OUTPUT_FOLDER = "output_step_b"

def solve_step_b():
    ensure_dir(OUTPUT_FOLDER)
    
    scene_paths = list_scene_paths(SCENE_FOLDER, "m*.png")
    model_paths = list_model_paths(MODEL_FOLDER)

    sift = create_sift()
    flann = create_flann()

    for scene_path in scene_paths:
        scene_name = os.path.basename(scene_path)
        print(f"\nProcessing Scene: {scene_name}")
        scene_rgb, final = detect_products_in_scene(
            scene_path,
            model_paths,
            sift,
            flann,
            verbose=True
        )
        if scene_rgb is None:
            continue

        for det in final:
            print(f"Product {det['name'].replace('.jpg','')} at {det['pos']}")
        
        draw_detections(scene_rgb, final)

        cv2.imwrite(os.path.join(OUTPUT_FOLDER, scene_name), scene_rgb)

if __name__ == "__main__":
    solve_step_b()