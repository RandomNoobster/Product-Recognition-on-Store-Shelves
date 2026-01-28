import numpy as np
import cv2
from matplotlib import pyplot as plt

def object_retrieve(img_query, img_train, min_match_count):
    # Split both images into Blue, Green, and Red channels
    q_chans = cv2.split(img_query)
    t_chans = cv2.split(img_train)
    
    total_good_matches = 0
    sift = cv2.SIFT_create()

    # Iterate through B, G, and R channels
    for i in range(3):
        # Detect and Compute for the current channel
        kp_q, des_q = sift.detectAndCompute(q_chans[i], None)
        kp_t, des_t = sift.detectAndCompute(t_chans[i], None)

        # Skip if one channel has no descriptors
        if des_q is None or des_t is None:
            continue

        # FLANN Matcher Setup
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des_q, des_t, k=2)

        # Ratio test to find good matches for this specific channel
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                total_good_matches += 1

    # Return total matches across all 3 channels
    if total_good_matches > min_match_count:
        return total_good_matches
    else:
        return -1

# --- Execution ---
path_scenes = "images/scenes/"
path_models = "images/models/"

# Load as color (1)
img_query = cv2.imread(path_scenes + 'e5.png', 1) 
imgs_train = ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '26.jpg', '25.jpg']

for path in imgs_train:
    img_train = cv2.imread(path_models + path, 1)
    if img_train is None: continue

    # Since we are checking 3 channels, you might need to increase 
    # the min_match_count (e.g., from 50 to 120)
    found = object_retrieve(img_query, img_train, 300)
    
    print(f"Found query object in {path}: {found > 0} (Matches: {found})")
    
    if found > 0:
        # Convert BGR to RGB for correct display in Matplotlib
        plt.imshow(cv2.cvtColor(img_train, cv2.COLOR_BGR2RGB))
        plt.show()