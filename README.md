# Product-Recognition-on-Store-Shelves

This repository contains our solution for the project work in the course `Computer Vision and Image Processing M`.

The images used are located in the `/images` folder, and the code is located in the `project.ipynb`-notebook. 

## Plan
### Step A
> Develop an object detection system to identify single instance of products given: one reference image for each item and a scene image. The system should be able to correctly identify all the product in the shelves image. One way to solve this task could be the use of local invariant feature as explained in lab session 5. 
- Preprocess the images?
    - What could be beneficial? Preprocessing the models, the scenes? What kind of preprocessing? Smoothing? 
- Use local invariant features of the models.  

### Step B
> In addition to what achieved at step A, the system should now be able to detect multiple instance of the same product. Purposely, students may deploy local invariant feature together with the GHT (Generalized Hough Transform). More precisely, rather than relying on the usual R-Table, the object model acquired at training time should now consist in vectors joining all the features extracted in the model image to their 
barycenter; then, at run time all the image features matched with respect to the model would cast votes for the position of the barycenter by scaling appropriately the  associated joining vectors (i.e. by the ratio of sizes between the matching features).
- Just do what we are told  

### Step C
> Try to detect as much products as possible in this challenging scenario: more than 40 different product instances for each picture, distractor elements (e.g. price tags…) and low resolution image. You can use whatever technique to achieve the maximum number of correct detection without mistake. 
- CNN would be useful but we dont have enough training data.
- Optionally we could look in pensum for some non-ML methods 