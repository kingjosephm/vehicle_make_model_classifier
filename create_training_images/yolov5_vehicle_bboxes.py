import torch
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from time import time
import os
import numpy as np
import argparse

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', None)

try:  # if run locally, not on GPU server
    import caffeine
    caffeine.on(display=True)
    skip = False
except ModuleNotFoundError:
    skip = True

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

    dims = np.array(results.imgs).shape
    dims = (dims[1], dims[2], dims[3])

    try:  # CPU implementation
        coordinates = results.xyxy[0].numpy()
    except TypeError:
        coordinates = results.xyxy[0].cpu().numpy()

    if len(coordinates) == 0:  # no objects found
        return [], ()

    # Keep only: 2 (cars), 5 (bus), 7 (truck)
    # Note - results.name is a list containing the string names and coordinates[:, -1] is the list index of that object
    coordinates = coordinates[(coordinates[:, -1] == 2.0) | (coordinates[:, -1] == 5.0) | (coordinates[:, -1] == 7.0)]

    # Keep only cars above given confidence threshold
    if len(coordinates[coordinates[:, 4] >= min_confidence]) > 0:
        coordinates = coordinates[coordinates[:, 4] >= min_confidence]
    else:  # none of car/bus/truck above confidence level
        return [], ()

    # Convert to bbox coordinates to int
    coordinates = np.concatenate((coordinates[:, :4].astype(int), coordinates[:, 4:]), axis=1)

    arr = coordinates[:, :4]  # bounding boxes themselves

    # Assume images well centered, so pick largest area object if >1 per image
    area = (arr[:, 3] - arr[:, 1]) * (arr[:, 2] - arr[:, 0])  # Format: xyxy
    coordinates = coordinates[np.argmax(area)]

    return coordinates[0:5].tolist(), dims


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-registry', type=str, help='path to CSV containing relative image paths and labels', required=True)
    parser.add_argument('--data', type=str, help='path to root directory where data located', required=True)
    parser.add_argument('--min-confidence', type=float, default=0.0, help='minimum confidence level of YOLOv5 bounding box object type')
    parser.add_argument('--yolo-model', type=str, default='small', choices=['small', 'medium', 'large', 'xl'], help='YOLOv5 model')
    parser.add_argument('--output', type=str, required=True, help='output path including title of file')
    args = parser.parse_args()
    assert (args.min_confidence >= 0 and args.min_confidence < 1), 'min-codence param is bounded 0-1!'
    return args


def main(opt):

    # Read in model, or download
    if opt.yolo_model == 'small':
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    elif opt.yolo_model == 'medium':
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
    elif opt.yolo_model == 'large':
        model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
    else:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

    df = pd.read_csv(opt.img_dir)

    start = time()

    output = []
    count = 0
    for i in df['Source Path']:

        coord, dims = detect_cars(os.path.join(opt.data, i), model, min_confidence=opt.min_confidence)
        output.append([coord, dims])
        if (count % 1000 == 0) and (count > 0):
            print(f"Cumulative time after {count} images: {time() - start:.2f} sec\n")
        count += 1
    temp = pd.DataFrame(output, columns=['Bboxes', 'Dims'])
    df = pd.concat([df, temp], axis=1)
    df['Dims'] = df['Dims'].apply(lambda x: [float(i) for i in x])  # convert to float to ensure CSV parses correctly

    df = df.sort_values(by=['Make', 'Model'])

    df.to_csv(opt.output, index=False)

    if skip:
        caffeine.off()


if __name__ == '__main__':

    opt = parse_opt()
    main(opt)