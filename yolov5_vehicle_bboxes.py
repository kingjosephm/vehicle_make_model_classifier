import torch
import pandas as pd
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from time import time
import caffeine

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', None)
caffeine.on(display=True)

def detect_cars(path, model, min_confidence=0.5):
    """
    :param path:
    :param model: yolov5 models.common.AutoShape object
    :param min_confidence: float, minimum confidence threshold for object type, range(0, 1)
    :return:
    """

    assert((min_confidence >=0) and (min_confidence<=1)), "Object type confidence level bounded strictly between 0-1!"

    results = model(path)  # applies NMS

    coordinates = results.xyxy[0].numpy()

    # Keep only: 2 (cars), 5 (bus), 7 (truck)
    # Note - results.name is a list containing the string names and coordinates[:, -1] is the list index of that object
    coordinates = coordinates[(coordinates[:, -1] == 2.0) | (coordinates[:, -1] == 5.0) | (coordinates[:, -1] == 7.0)]

    # Keep only cars above given confidence threshold
    coordinates = coordinates[coordinates[:, 4] >= min_confidence]
    return coordinates[:, 0:5].tolist()



if __name__ == '__main__':

    # Read in model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Read in pd.DataFrame
    path = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier'
    df = pd.read_csv(os.path.join(path, 'Vehicle Make Model Directory.csv'))

    # Make path absolute
    df['Path'] = df['Path'].apply(lambda x: os.path.join(path, 'data', x))

    start = time()

    output = []
    count = 0
    for i in df['Path']:
        output.append(detect_cars(i, model, min_confidence=0.5))
        if (count % 1000 == 0) and (count > 0):
            print(f"Cumulative time after {count} images: {time() - start:.2f} sec\n")
        count += 1
    df['Bboxes'] = pd.Series(output)

    # Make path again relative
    df['Path'] = df.Path.str.split('/').apply(lambda x: x[-4:]).apply(lambda x: '/'.join(x))

    df.to_csv('Vehicle Make Model Directory Bboxes.csv', index=False)  # Output to working dir

    caffeine.off()