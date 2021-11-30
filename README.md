# Introduction
This code develops a large (n=690,014) dataset of contemporary passenger motor vehicles in the U.S. and an image classifier model to identify these vehicle makes and models. The end product of this code are model weights, which are used in a separate ML edge pipeline. A mockup of this pipeline can be found [here](https://github.boozallencsn.com/MERGEN/vehicle_image_pipeline).

# Training Data
## Dataset construction
#### Sampling frame
- To create a representative sample of vehicle make and model images for the U.S. passenger vehicle market we rely on the [back4app.com](https://www.back4app.com/database/back4app/car-make-model-dataset) database, an open-source dataset providing detailed information about motor vehicles manufactured in the US between the years 1992 and 2022. 
<br> <br />
- A copy of this database are stored locally at `./data/make_model_database.csv` along with the script to generate this extract at `./create_training_images/get_make_model_db.py`. At the time the data were queried, this database contained information on vehicles up through and including 2022 models, though 2022 models are only available for some manufacturers. The database contained information on 59 distinct vehicle manufacturers and 1,032 detailed make-model combinations over the period. 
<br> <br />
- We drop 4 small vehicle manufacturers (e.g. Fisker, Polestar, Panoz, Rivian), 8 exotic car manufacturers (e.g. Ferrari, Lamborghini, Maserati, Rolls-Royce, McLaren, Bentley, Aston Martin, Lotus), and 7 brands with sparse information in the dataset (e.g. Alfa Romeo, Daewoo, Isuzu, Genesis, Mayback, Plymouth, Oldsmobile), reducing the number of distinct vehicle manufacturers in the data to 40. 
<br> <br />
- The resulting 40 manufacturers, their years present in the database, and the number of aggregated models per manufacturer in the database (see description below) are displayed in the following table.

| Manufacturer | Years in Database | Number of Models |
| --------- | ----- | ------- |
| Acura | 2000-2022 | 13 
| Audi | 2000-2021 | 26
| BMW | 2000-2021 | 27
| Buick | 2000-2021 | 14
| Cadillac | 2000-2021 | 19
| Chevrolet | 2000-2022 | 38
| Chrysler | 2000-2021 | 14
| Dodge | 2000-2021 | 18
| Fiat | 2012-2021 | 2
| Ford | 2000-2021 | 28
| GMC | 2000-2022 | 11
| HUMMER | 2000-2010 | 4
| Honda | 2000-2022 | 17
| Hyundai | 2000-2022 | 18
| INFINITI | 2000-2021 | 17
| Jaguar | 2000-2021 | 10
| Jeep | 2000-2022 | 9
| Kia | 2000-2022 | 19
| Land Rover | 2000-2021 | 6
| Lexus | 2000-2021 | 15
| Lincoln | 2000-2021 | 15
| MINI | 2002-2020 | 8
| Mazda | 2000-2021  | 18
| Mercedes-Benz | 2000-2022 | 28
| Mercury | 2000-2011 | 11
| Mitsubishi | 2000-2022 | 11 
| Nissan | 2000-2022  | 20
| Pontiac | 2000-2010 | 15
| Porsche | 2000-2021 | 11
| RAM | 2011-2021 | 4
| Saab | 2000-2011 | 5
| Saturn | 2000-2010 | 9
| Scion | 2004-2016 | 8
| Subaru | 2000-2022  | 12
| Suzuki | 2000-2013 |  12
| Tesla | 2012-2021 | 3
| Toyota | 2000-2021  | 24 
| Volkswagen | 2000-2022 | 18 
| Volvo | 2000-2021 | 16
| smart | 2008-2018 | 1

- To reduce the number of detailed vehicle make-model combinations, related  models are combined together (e.g. Ford F-150 Super Cab and Ford F350 Super Duty Crew Cab are combined into a single Ford F-Series category) using the script at `./create_training_images/restrict_population_make_models.py`. This reduced the number of unique make-model combinations over the period to **574**. 
<br> <br />
- The restricted vehicle database is stored at `./data/make_model_database_mod.csv` with a corresponding analysis of this database in `./create_training_images/back4app_database_analysis.ipynb`. A full list of these 574 make-model classes can be seen by scrolling down.

#### Sampling method
- Having defined the population of vehicles of interest, we scrape Google Images to download images that will be used as our training dataset. To capture sufficient variation *within* each vehicle make-model combination over time we scrape images using the detailed vehicle model descriptor, combined with the vehicle category (e.g. coupe, sedan, hatchback, SUV, comvertible, wagon, van, pickup), for every year available. In told, this produced 8,274 unique make-(detailed-)model-category-year combinations.
<br> <br />
- For every make-(detailed-)model-category-year combination, we scrape 100 images, which typically results in 85-90 savable JPG images. We store these data in separate directories on disk based on make-(aggregated-)model-year. In each directory, approximately 95% of saved images are exterior vehicle photographs with the vast majority corresponding to the correct vehicle make, model and year. 

#### Sample restrictions
- A full analysis of the scraped image dataset in the notebook at `./create_training_images/scraped_image_analysis.ipynb`. 690,014 total images were scraped for all 574 make-model classes over the period. Of these, 656,294 (95.11%) images were identified as having a vehicle object in them, according to the YOLOv5 (XL) algorithm. Specifically, we restict to abjects that this algorithm labels as a car, truck, or bus and with a confidence of >= 0.75. If multiple such images are identified in a particular image, we keep the one with the largest bounding box area. In the next section we present a kernel density plot of the distribution of YOLOv5 XL confidence levels in our training data.
  - Auxiliary analyses indicated that vehicle make-model classifier model performance was similar using a 0.5 YOLOv5 XL confidence level, though we opted for a higher threshold to err on the side of caution.
  <br> <br />
- To ensure our training set contains adequately-sized images, we further restrict to images whose bounding boxes are > 1st percentile of pixel area, which reduced the total image count to 649,731 (94.16% of original images). The 1st percentile corresponded to 8,911 pixels, or approximately a 94 x 94 pixel image, which is comparably small. 
  - Auxiliary analyses indicated that increasing this minimum object size threshold did not appreciably enhance model performance, while also reducing the number of sample images.
  <br> <br />
- In the notebook at `./create_training_images/compare_yolov5_models.ipynb` we examine the distributons of confidence and bounding box area across the YOLOv5 small, medium, large, and XL models. 

## Descriptives
- A kernel density plot of YOLOv5 XL bounding box confidence.
<br />

![kde_bb_confidence](./create_training_images/kdeplot_yolo_confidence.png)

- The empirical cumulative distribution function (ECDF) of bound box area from the YOLOv5 XL model.
<br />

![ECDF_Bbox](./create_training_images/ecdf_bounding_box_area.png)

- In supplementary analyses we imposed restrictions on the minimum image count per class, meaning make-model classes below this threshold were excluded from training and evaluation. This, however, had little impact on model performance; correspondingly, we include all 574 classes in our final model.
<br> <br />
- The table and figure below display key statistical moments and the distribution in the number of images per class, respectively, net of our analytic restrictions.

| Statistic | Value |
| --------- | ----- |
| Classes   | 574   |
| Mean      | 889.75 |
| std       | 980.07 |
| min       | 56.00  |
| 5%        | 109.00 |
| 10%       | 149.00 |
| 25%       | 287.75 |
| 50%       | 557.00 |
| 75%       | 1117.75 |
| 90%       | 1908.40 |
| 95%       | 7821.00 |
| max       | 7821.00 |

<br />

![ECDF_img_count](./create_training_images/ecdf_img_count.png)



- The following figure illustrates the final number of images per make-model class in our resulting training data, net of analytic restrictions.
<br> <br />

![test](./create_training_images/final_img_count_class.png)


# Pipeline to Collect Training Images

The following scripts were run in this order to create the sample of training images:

  1) `./create_training_images/get_make_model_db.py`, which queries the back4app database, outputting -> `./data/make_model_database.csv`.
<br> <br />
  2) `./create_training_images/restrict_population_make_models.py`, which standardizes and fixes some errors in vehicle makes and models, outputting -> `./data/make_model_database_mod.csv`.
<br> <br />
  3) `./create_training_images/scrape_vehicle_make_models.py`, which scrapes Google Images for each detailed make-model-year combination.
<br> <br />
  4) `./create_training_images/create_image_directory.py`, which ensures non-duplicate and valid URLs and creates the image dataframe that contains a path and label to each JPG image. This outputs -> `./data/MakeModelDirectory.csv`.
<br> <br />
  5) `./create_training_images/yolov5_vehicle_bboxes.py`, which classifies objects in images using [YOLOv5](https://github.com/ultralytics/yolov5). This outputs -> `./data/Bboxes.csv`.

# Pipeline to Train the Make-Model Classifier
We develop code locally on our local laptop and upload updated scripts to the GPU cluster to execute code. To upload the scripts that run the make-model classifier, `MakeModelClassifier.py` and `core.py`, along with the image directory `./data/Bboxes.csv` enter in your console:

    sh driver.sh

Code will be copied onto the working directory for your profile on the GPU in a directory called `scripts`.


### Data on the GPU cluster
The training data described above are stored in a Docker volume called `MERGEN_Make_Model_data`. Images are also stored outside of Docker on the cluster at `/home/kingj/scraped_images`.

### Model output
Results from running `MakeModelClassifier.py` are stored in a Docker volume called `MERGEN_Make_Model_output`.

### CSV file to associate images with labels
A CSV file containing paths to each JPG image, YOLOv5 bounding box coordinates, make-model class labels, and image dimensions is stored locally at `./data/Bboxes.csv`. At training time we use this CSV to link each image (a string path, in this dataframe) to its associated label and bounding box coordinates. ***We do not pre-crop images down to their YOLOv5 bounding box; instead, images are cropped as they are streamed in the training process***. The make-model classifier is trained using cropped cropped images, though we dilate these bounding boxes by 5px as YOLOv5 bounding boxes tend to be tightly cropped.

### Run the classifier
To run the classifier using an ResNet50 layer, for example, in a detached Docker container, enter:

    docker run -it \
        --name make_model_classifier \
        --rm -d \
        --mount type=bind,source=/home/kingj/scripts,target=/scripts \
        --mount source=MERGEN_Make_Model_data,target=/data \
        --mount source=MERGEN_Make_Model_output,target=/output \
        --gpus device=GPU-7a7c102c-5f71-a0fd-2ac0-f45a63c82dc5 \
        king0759/tf2_gpu_jupyter_mpl:v3 python3 ./scripts/MakeModelClassifier.py \
        --train --data=/data --epochs=130 --output=/output --logging='true' \
        --save-weights='true' --dropout=0.25 --patience=10 --batch-size=256 \
        --units2=4096 --units1=2048 --model='resnet' --resnet-size='50' \
        --min-class-img-count=0 --learning-rate=0.0001 --optimizer='adam'

The Docker image `king0759/tf2_gpu_jupyter_mpl:3` contains all the dependent modules. Importantly, you will need to update the mount source above to the scripts directory on your working directory, if you make changes to these scripts.You should also select one or more GPU to designate for your container:

- GPU-8121da2f-b1c3-d231-a9ab-7d6f598ba2dd
- GPU-7a7c102c-5f71-a0fd-2ac0-f45a63c82dc5
- GPU-0c5076b3-fe4a-0cd8-e4b7-71c2037933c0)
- GPU-3c51591d-cfdb-f87c-ece8-8dcfdc81e67a

These are the unique identifiers for GPUs 0-3, respectively. To see which GPUs are free and which are being used type `nvidia-smi`. Their listed order corresponds to the above.

# Results
### Best performing model

- Framework: TensorFlow Keras
- Optimizer: Adam
- Batch size: 256
- ResNet50V2
- GlobalAveragePooling2D
- Dropout rate: 0.2
- Dense layers: 4096 x 2048 units
- Bounding box dilation: 5px
- Max training epochs: 130
- Early stopping patience: 10 epochs
- Learning rate: 0.0001
- YOLOv5 confidence threshold: 0.5
- Minimum YOLOv5 bounding box area: 3,731 pixels (5th percentile)
- Minimum training images per class: 0
- Total classes: 574

![accuracy](./results/Accuracy.png)
![loss](./results/Loss.png)


- A more extensive analysis of performance in the test set can be viewed in the notebook at `./results/TestSetAnalysis.ipynb`.
<br> <br />
- The following table contains results from a host of recent experiments to find the optimal model given our training data. In particular number 12 is our best performing model.

| Number | Pretrained Model | # Classes | Dense Layers | Dropout Rate | Test Argmax(0) | Test Argmax(0:5) | Stanford Argmax(0) | Stanford Argmax(0:5) |
| ------ | ---------------- | --------- | ------------ | ------------ | -------------- | ---------------- | ------------------ | -------------------- |
|1 | Xception | 552 | 512 | 0.05 | 0.4409 | 0.7184 |
|2 | ResNet101V2 | 552 | 512 | 0.05 | 0.6031 | 0.8406 |
|3 | ResNet101V2 | 574 | 512 | 0.05 | 0.6027 | 0.8365 |
|4 | ResNet101V2 | 477 | 512 | 0.05 | 0.6192 | 0.8495 |
|5 | ResNet101V2 | 352 | 512 | 0.05 | 0.6348 | 0.8637 |
|6 | Inception | 574 | 512 | 0.05 | 0.4616 | 0.7272 |
|7 | ResNet152V2 | 574 | 512 | 0.05 | 0.6113 | 0.8440 | 
|8 | ResNet101V2 | 574 | 1024 | 0.05 | 0.6282 | 0.8499 |
|9 | ResNet101V2 | 574 | 1024 x 1024 | 0.05 | 0.6431 | 0.8679 |
|10 | MobileNetV2 | 574 | 1024 x 1024 | 0.05 | 0.4277 | 0.7089 |
|11 | ResNet50V2 | 574 | 2048 x 1024 | 0.1 | 0.6614 | 0.8756 | 0.5996 | 0.8218 |
|12 | ResNet50V2 | 574 | 4096 x 2048 | 0.2 | 0.6896 | 0.8887 | 0.5918 | 0.8253 |
|13 | ResNet50V2 | 574 | 8192 x 4096 | 0.25 | 0.6804 | 0.8834 | 0.5900 | 0.8211 |

All models trained using the Adam optimizer, a learning rate of 0.0001, max epochs of between 130-200 epochs with early stopping after 10 epochs, a minimum YOLOv5 bounding box area of 3,731 pixels, YOLOv5 confidence of 0.5, and batch size of 256.

### External validity
We employ a second set of test images from the [Stanford car dataset](https://www.kaggle.com/jessicali9530/stanford-cars-dataset) to evaluate the generalizability of our model. This dataset contains 16,185 images and 196 classes, though only 124 classes overlap with our scraped images. The Stanford images are likewise dated, with the newest make-model being from 2012. Nonetheless, our model performs comparably with these data as with an unseen test subset from our original data (see table above). The CSV image dataframe for these data was curated via `./create_test_images/curate_stanford_img_dir.py`.