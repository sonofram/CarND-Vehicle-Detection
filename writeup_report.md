---

# **Vehicle Detection Project** #

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video_out.mp4



## Summary ##

Vehicle detection in done two seperate steps. **First step** involves supervised learning and followed by **Second step** of real-time vehicle detection based on the trained supervised learning model. 

---

### Step#1 Supervised learning involves ###

1. feature extraction 
2. model identificaiton 
3. training model
4. testing model for accuracy

**Feature Extaction** : Three different type of features that were extracted in this vehicle detection. Those are

a. Color Histogram features - Differentiating images by the intensity and range of color they contain can be helpful for looking at car vs non-car images.
b. Spatial binning of Color - extract pixels from vehicle and non-vehicle that wil help in seperating them out.
c. Histogram of Oriented Gradients(HOG) - extract gradient of image.

**Model Identification** : Many model were tried and Linear SVC turned out to be suitable for vehicle detection.

```python

svc = LinearSVC()

```

**Training Model** : To train the model, all sample dataset directories were explored and images of car and non-car are used in training.

```python
veh_dir = ['./vehicles/GTI_Far','./vehicles/GTI_Left','./vehicles/GTI_MiddleClose','./vehicles/GTI_Right',
           './vehicles/KITTI_extracted']
non_veh_dir = ['./non-vehicles/GTI','./non-vehicles/Extras']

img_ext = '.png'

```

Following parameters are set for all images processed for feature extraction.

```python
colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
scale = 1.0
histbins = 32
spatialsize = (32,32)
```

All car and non-car images are processed for feature extraction. Each image processed thru function **single_img_features()** and this specific function extracted all features mentioned above and give consolidated list of features for each image. 

```python
car_features = []
notcar_features = []
for car in cars_dir:
    image = mpimg.imread(car)
    car_feature = single_img_features(img = image, color_space=colorspace, spatial_size=spatialsize, \
                        hist_bins=histbins, orient=orient, pix_per_cell=pix_per_cell, \
                        cell_per_block=cell_per_block, hog_channel=hog_channel, \
                        spatial_feat=True, hist_feat=True, hog_feat=True, \
                        y_start_stop = [None,None], x_start_stop = [None,None], scale=scale)

    car_features.append(car_feature)

for notcar in notcars_dir:
    image = mpimg.imread(notcar)
    notcar_feature = single_img_features(img = image, color_space=colorspace, spatial_size=spatialsize, \
                        hist_bins=histbins, orient=orient, pix_per_cell=pix_per_cell, \
                        cell_per_block=cell_per_block, hog_channel=hog_channel, \
                        spatial_feat=True, hist_feat=True, hog_feat=True, \
                        y_start_stop = [None,None], x_start_stop = [None,None], scale=scale)

    notcar_features.append(notcar_feature)

print("Number of car features: ",len(car_features)," Number of features in each car: ",np.array(car_features[0]).shape)    
print("Number of notcar features: ",len(notcar_features)," Number of features in each notcar: ",np.array(notcar_features[0]).shape)    

```

**Number of car images:   8792  Number of notcar images:  8968** 

**Number of car features:  8792  Number of features in each car:  (8460,)** 

**Number of notcar features:  8968  Number of features in each notcar:  (8460,)**

All extracted features are scaled for ZERO mean and also label were generated for car features and non-car features.

```python
#Scale images between 0 mean    
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
```

Finally, features are split into training dataset and testing dataset. Ration for train/split is 80/20.

```python
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()

```

**Testing Model for Accuracy** : Test accuracy also measured

```python

print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

```
**Using: 9 orientations 8 pixels per cell and 2 cells per block**

**Feature vector length: 8460**

**6.77 Seconds to train SVC...**

**Test Accuracy of SVC =  0.9924**


Also, above trained model will be persisted, so that, it can re-used in vehicle detection.

```python
save_path = "./svm_training.p"

svm_train = {  'colorspace': colorspace,
               'orient': orient,
               'pix_per_cell': pix_per_cell,
               'cell_per_block': cell_per_block,
               'hog_channel': hog_channel,
               'scale': scale,
               'histbins' : histbins,
               'spatialsize' : spatialsize,
               'svc' : svc,
               'scaler' : X_scaler
              }

with open(save_path, 'wb') as f:
    pickle.dump(svm_train, file=f)
```


---





### Step#2 Vehicle Detection ###

Vehicle detection pipeline can utilize above model to identify vehicles in images. Following steps involved in vehicle identification pipeline


Persisted model will be read for vehicle identificaiton.

```python

save_path = "./svm_training.p"
svc_pickle = pickle.load( open( save_path, "rb" ) )

colorspace = svc_pickle['colorspace'] 
orient = svc_pickle['orient']
pix_per_cell = svc_pickle['pix_per_cell']
cell_per_block = svc_pickle['cell_per_block']
hog_channel = svc_pickle['hog_channel']
scale = svc_pickle['scale']
histbins = svc_pickle['histbins']
spatialsize = svc_pickle['spatialsize']
svc = svc_pickle['svc']
X_scaler = svc_pickle['scaler']

```

Image that need to be processed will be sent through function "process_img()". This specific method does following steps to detect vehicle in image.

1. Convert image to same color space as used in model. In this pipeline **YUV** color space is utlized.
2. For HOG calculation will be done on below half of image instead of entire image for better performance. Even in Below half of image also processed based on HOG Sub-sampling approach. HOG Sub-sampling will be explained in more detail in below sections.  
3. At high level, image is broken down into cells(8x8, consider it as scale = 1.0) and this small section will be processed for feature extraction.   
4. Color features, spatial binning features and HOG features will be extracted.
5. These features will be sent to SVC model for prediction for car/non-car. If identified as car, rectangular coordinates will be identified and drawn on the image to mark the identification of car on image.
5. Above steps are repeates with different cell scales(i.e. 8x8 size with scale =1.5, 2.0, 2.5 ..etc). This is done primarily to capture the cars that are near mostly likely to get marked with scale = 2.0 ro 2.5 and cars that are far might get captured by scale = 1.0
6. There might lot of false-positives, therefore, heat map technique is applied to filter out false-positives. This technique is explained in more detail in below sections.
7. In case of video processing for vehicle identification, heat map technique involves capturing image identification rectangular boxes for few previous frames and applying heat map to retain only those rectangular boxes that are consistently present in all previous frames.

---

## Details ##

###Histogram of Oriented Gradients (HOG)

HOG features are extracted twice. 
 1. During model training process
 2. Vehicle identificaiton pipeline.
 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]


####HOG during model training####

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.
Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. During training HOG is calculate on entire image and this is not the case while calculating HOG in **vehicle identification pipeline**


![alt text][image2]


####HOG in Vehicle identification pipeline####

Here HOG is not calculated on entire image, instead, only calculated for second half of the image.

---

###Sliding Window Search

Sliding window search is techinque used to search of cars/vehicles in the image. In this techinque, area of interest in image will be searched by splitting it into multiple cell blocks. Some of important parameters considered in this technique are:

**Pixes_per_cell** -  Number pixels considered in a cell.
**window** - Number of cell considered. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%. 
**scale** - This is important parameter while identifying far and near vehicles. window of scale = 1(as mentioned above) is more suitable for vehicles that are far and scales like 1.5,2.0,2.5..etc are better for identifying nearer vehicles.


![alt text][image3]

####Pipeline for searching vehicle

Few important functions are explained below before explaing pipeline

**process_img** : Main function that drives the pipeline. This function applies sliding window search and applies it in six different scales. These scaled windows are applied in different Y-axis ranges as shown below

```python

y_start_end =[(380,480), \
                 (400,480), \
                 (380,580), \
                 (400,580), \
                 (380,680), \
                 (400,680), \
                 (380,680), \
                 (400,680)]
    scale_list = [1.0,1.0,1.5,2.0,2.5,3.5]
    
    for i in range(len(scale_list)):
        ystart = y_start_end[i][0]
        ystop =  y_start_end[i][1]
        scale = scale_list[i]
        out_img,box_list_comp = find_cars(img, xstart,xstop, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatialsize, histbins,colorspace)    
        if len(box_list_comp) > 0:
            box_list.append(box_list_comp)

```

These Y-axis ranges are area of interest in image that will be split into windows and searched for vehicle. For each scale, **find_cars** function is called and details of this function are mentioned below.  **find_cars** return the rectangular box listing that were identified by svc model as vehicle identification. Theese rectangular boxes will be passed to heatmap technique to avoid false positives.

```python

      	# Add heat to each box in box list
        heat = add_heat(heat,box_list)
        #print("Head: ",heat[heat > 0])    
        # Apply threshold to help remove false positives
        # Heat threshold applied to 2
        heat = apply_threshold(heat,2)
        #print("heat after threshold: ",heat[heat> 0])
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        #print(labels[1], 'cars found')
        #plt.imshow(labels[0], cmap='gray')
        draw_img = draw_labeled_bboxes(np.copy(img), labels,xstart)

```

**find_cars** : Even though **process_img** is main function, heavy lifing is done by find_cars. This method actually applies sliding window technique. It also extracts features from each window and predicts whether it is vehicle or not. Further, all vehicles identified are captures in images in rectangular coordinates.

**add_heat** : Method mainly to add heatmap to pixels within rectangular coordinates and this is primarily associated with thresholds to filter out false positives.

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

**apply_threshold** : Method that filters out rectangular coordinates that does not meeting specified thresholds. 

***Thresholding was defined as 2 while processing single images while testing. however, for video processing, thresholding was dynamic. Rectangular boxes that identify vehicles are capture along with few previous frames. Only when rectangular boxes that are consistently available accross previous capture frames will be retained and marked as vehicle.***

To handle this new python class **CarsDetected** has been designed. This car stores previous frames from video based on initial frame thresholds(default is to store previous 10 frames). 


**draw_labeled_bboxes** : find the min/max corners be consilidating all rectangular coordinates in specific point to come out with single rectangular box to mark the vehicle.

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---

### Video Implementation

Here's a [link to my video result](./project_video_out.mp4)

---

###Discussion


1. SVC model was trained using few images. Probably, it might have to trained more. At the same time, there is no exception bucket. It is always either car or non-car. There should be some thresholding defined that can be none(neither car nor non-car). In this case, these can be pulled into analysis stage for further improving the model. Training also should be done in different weather patterns like (raining, foggy, snowing etc).

2. While applying sliding window technqiue, we apply different scales of windows and there is optimized scale that was applied. I have applied scales like 1.0, 1.5, 2.0 and 2.5. Need some more experimentation to identify optimization.

