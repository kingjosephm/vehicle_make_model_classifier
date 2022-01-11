# Table of contents
- [Dataset construction](#dataset-contruction)
  - [Sampling frame](#sampling-frame)
  - [Manufacturers in data](#manufacturers-in-data)
  - [Sampling method](#sampling-method)
  - [Analytic restrictions](#analytic-restrictions)
- [Pipeline to create training image dataset](#pipeline-to-create-training-image-dataset)
- [Analytic sample descriptive statistics](#analytic-sample-descriptive-statistics)
  - [Distribution of images per class](#distribution-of-images-per-class)
  - [Images per make-model](#images-per-make-model)


## Dataset construction
#### Sampling frame
- To create a representative sample of vehicle make and model images for the U.S. passenger vehicle market we rely on the [back4app.com](https://www.back4app.com/database/back4app/car-make-model-dataset) database, an open-source dataset providing detailed information about motor vehicles sold in the US between the years 1992 and 2022. 
<br> <br />
- A copy of this database are stored locally at `./data/make_model_database.csv` along with the script to generate this extract at `./create_training_images/get_make_model_db.py`. At the time the data were queried, this database contained information on vehicles up through and including 2022 models, though 2022 models are only available for some manufacturers. The database contained information on 59 distinct vehicle manufacturers and 1,032 detailed make-model combinations over the period. 
<br> <br />
- We drop 4 small vehicle manufacturers (e.g. Fisker, Polestar, Panoz, Rivian), 8 exotic car manufacturers (e.g. Ferrari, Lamborghini, Maserati, Rolls-Royce, McLaren, Bentley, Aston Martin, Lotus), and 7 brands with sparse information in the dataset (e.g. Alfa Romeo, Daewoo, Isuzu, Genesis, Mayback, Plymouth, Oldsmobile), reducing the number of distinct vehicle manufacturers in the data to 40. 
<br> <br />
- The resulting 40 manufacturers, their years present in the database, and the number of aggregated models per manufacturer in the database (see description below) are displayed in the following table.

### Manufacturers in data

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

### Sampling method
- Having defined the population of vehicles of interest, we scrape Google Images to download images that will be used as our training dataset. To capture sufficient variation *within* each vehicle make-model combination over time we scrape images using the detailed vehicle model descriptor, combined with the vehicle category (e.g. coupe, sedan, hatchback, SUV, convertible, wagon, van, pickup), for every year available. In told, this produced 8,274 unique make-(detailed-)model-category-year combinations.
<br> <br />
- For every make-(detailed-)model-category-year combination, we scrape 100 images, which typically results in 85-90 savable JPG images. We store these data in separate directories on disk based on make-(aggregated-)model-year. In each directory, approximately 95% of saved images are exterior vehicle photographs with the vast majority corresponding to the correct vehicle make, model and year. 

### Analytic restrictions
- A full analysis of the scraped image dataset in the notebook at `./create_training_images/scraped_image_analysis.ipynb`. 664,678 total images were scraped for all 574 make-model classes over the period. Of these, 631,973 (95.08%) images were identified as having a vehicle object in them, according to the YOLOv5 (XL) algorithm. 
 <br> <br />
- To ensure fidelity of our resulting vehicles images and bounding box coordinates (for cropping), we restrict to bounding box coordinates from the YOLOv5 XL model with a confidence >= 0.5 (609,265 images; 91.66%). If multiple such images are identified in a particular image, we keep the one with the largest bounding box area. In the next section we present a kernel density plot of the distribution of YOLOv5 XL confidence levels in our training data.
  - We considerably examined the impact of YOLOv5 model confidence level on vehicle classifier performance and found a weak correlation. While the particular YOLOv5 model selected (e.g. small/medium/large/XL) affected classifier performance by about +/- 3 percentage points, varying the level of confidence for the same YOLOv5 model did not greatly affect the classifier.
  <br> <br />
- To ensure our training set contains adequately-sized images, we further restrict to images whose bounding boxes are > 1st percentile of pixel area, which reduced the total image count to 603,899 (90.86% of original images). The 1st percentile corresponded to 8,911 pixels, or approximately a 94 x 94 pixel image, which is comparably small. 
  - Auxiliary analyses indicated that increasing this minimum object size threshold did not appreciably enhance model performance, while also reducing the number of sample images.
  <br> <br />
- In the notebook at `./create_training_images/compare_yolov5_models.ipynb` we examine the distributons of confidence and bounding box area across the YOLOv5 small, medium, large, and XL models. 


# Pipeline to create training image dataset

The following scripts were run in this order to create the sample of training images:

  1) `./create_training_images/get_make_model_db.py`: queries the back4app database, outputting `./create_training_images/data/make_model_database.csv`
<br> <br />
  2) `./create_training_images/restrict_population_make_models.py`: standardizes and fixes some errors in vehicle makes and models, outputting `./create_training_images/data/make_model_database_mod.csv`
<br> <br />
  3) `./create_training_images/scrape_vehicle_make_models.py`: scrapes Google Images for each detailed make-model-year combination
<br> <br />
  4) `./create_training_images/create_image_directory.py`: ensures non-duplicate and valid URLs and creates the image dataframe that contains a path and label to each JPG image. Outputs image registry
<br> <br />
  5) `./create_training_images/yolov5_vehicle_bboxes.py`: detects objects in images using [YOLOv5](https://github.com/ultralytics/yolov5) algorithm


# Analytic sample descriptive statistics

### Kernel density plot of YOLOv5 XL bounding box confidence

![kde_bb_confidence](create_training_images/figs/kdeplot_yolo_confidence.png)

### Empirical cumulative distribution function (ECDF) of bound box area from the YOLOv5 XL model

![ECDF_Bbox](create_training_images/figs/ecdf_bounding_box_area.png)

In supplementary analyses we imposed restrictions on the minimum image count per class, meaning make-model classes below this threshold were excluded from training and evaluation. This, however, had little impact on model performance; correspondingly, we include all 574 classes in our final model.

### Distribution of images per class
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

### ECDF of image count
![ECDF_img_count](create_training_images/figs/ecdf_img_count.png)

### Images per make-model
![test](create_training_images/figs/final_img_count_class.png)