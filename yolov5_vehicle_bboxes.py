import torch
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from time import time
import caffeine
import os
import numpy as np

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', None)
caffeine.on(display=True)

def detect_cars(path, model, min_confidence=0.5):
    """
    Uses YOLOv5 to detect objects, keeping only cars/buses/trucks, restricts to bounding boxes above confidence
    threshold, keeps object with largest area according to bounding box coordinates.
    :param path: absolute path to jpg/png image
    :param model: yolov5 models.common.AutoShape object
    :param min_confidence: float, minimum confidence threshold for object type, range(0, 1)
    :return: list, where elements 0-4 are xyxy bounding box coordinates and the 5th element is the confidence
    """

    assert((min_confidence >= 0) and (min_confidence <= 1)), "Object type confidence level bounded strictly between 0-1!"

    try:
        results = model(path)  # applies NMS
    except ValueError:  # Image is blank
        return []

    coordinates = results.xyxy[0].numpy()

    if len(coordinates) == 0:  # no objects found
        return []

    # Keep only: 2 (cars), 5 (bus), 7 (truck)
    # Note - results.name is a list containing the string names and coordinates[:, -1] is the list index of that object
    coordinates = coordinates[(coordinates[:, -1] == 2.0) | (coordinates[:, -1] == 5.0) | (coordinates[:, -1] == 7.0)]

    # Keep only cars above given confidence threshold
    if len(coordinates[coordinates[:, 4] >= min_confidence]) > 0:
        coordinates = coordinates[coordinates[:, 4] >= min_confidence]
    else:  # none of car/bus/truck above confidence level
        return []

    # Convert to bbox coordinates to int
    coordinates = np.concatenate((coordinates[:, :4].astype(int), coordinates[:, 4:]), axis=1)

    arr = coordinates[:, :4]  # bounding boxes themselves

    # Assume images well centered, so pick largest area object if >1 per image
    area = (arr[:, 3] - arr[:, 1]) * (arr[:, 2] - arr[:, 0])  # Format: xyxy
    coordinates = coordinates[np.argmax(area)]

    return coordinates[0:5].tolist()

if __name__ == '__main__':

    # Read in model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    path = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier/data_directories'

    df = pd.read_csv(os.path.join(path, 'MakeModelDirectory.csv'))

    start = time()

    output = []
    count = 0
    for i in df['Source Path']:
        output.append(detect_cars(i, model, min_confidence=0.5))
        if (count % 1000 == 0) and (count > 0):
            print(f"Cumulative time after {count} images: {time() - start:.2f} sec\n")
        count += 1
    df['Bboxes'] = pd.Series(output)

    df = df.sort_values(by=['Make', 'Model', 'Year'])

    df.to_csv(os.path.join(path, 'MakeModelDirectory_Bboxes.csv'), index=False)

    caffeine.off()