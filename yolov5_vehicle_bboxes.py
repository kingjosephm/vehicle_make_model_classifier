import torch
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from time import time
import caffeine
import os

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

def compile_image_directory(rootDir):

    lst = []
    for subdir, dirs, files in os.walk(rootDir):
        for file in [i for i in files if 'jpg' in i or 'png' in i]:
            lst.append('/'.join(os.path.join(subdir, file).split('/')[-4:]))  # does not count empty subdirectories
    df = pd.DataFrame(lst, columns=["Orig Path"])
    df['Make'] = df['Orig Path'].apply(lambda x: x.split('/')[0])
    df['Model'] = df['Orig Path'].apply(lambda x: x.split('/')[1])
    df['Year'] = df['Orig Path'].apply(lambda x: x.split('/')[2]).astype(int)
    df['Orig Path'] = '/'.join(rootDir.split('/')[:-1]) + '/' + df['Orig Path']

    return df[['Make', 'Model', 'Year', 'Orig Path']].sort_values(by=['Make', 'Model', 'Year']).reset_index(drop=True)


if __name__ == '__main__':

    # Read in model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    rootDir = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier/scraped_images'

    df = compile_image_directory(rootDir)

    start = time()

    output = []
    count = 0
    for i in df['Orig Path']:
        output.append(detect_cars(i, model, min_confidence=0.5))
        if (count % 1000 == 0) and (count > 0):
            print(f"Cumulative time after {count} images: {time() - start:.2f} sec\n")
        count += 1
    df['Bboxes'] = pd.Series(output)

    outputDir = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier/data_directories'

    df.to_csv(os.path.join(outputDir, 'MakeModel_Bboxes.csv'), index=False)

    caffeine.off()