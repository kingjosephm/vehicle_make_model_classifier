import pandas as pd
import cv2
import os
import ast
import numpy as np

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', None)


def main(df, outputDir, resize=400, min_area=50625, min_confidence=0.5):
    """
    Loops through all image paths in dataframe `df`, restricts to bounding box coordinates in `df['Bboxes']`,
        resizes image and outputs to directory.
    :param df: pd.DataFrame
    :param outputDir: str, path to output each image
    :param resize: int, w x h pixel dimensions of final image
    :param min_area: int, minimimum w x h pixel dimensions for a vehicle to be output, default 225*225
    :param min_confidence: float, minimum confidence threshold of vehicle output
    :return:
    """

    for x in range(len(df)):

        # Read in single image
        foo = df.iloc[x]
        img = cv2.imread(foo['Path'])

        # Skip entire image if none of object's confidence above threshold
        if (np.array(foo['Bboxes'])[:, 4] < min_confidence).all():
            continue
        else: # restrict to object above confidence thresh
            arr = np.array(foo['Bboxes'])
            arr = arr[arr[:, 4] >= min_confidence][:, :4].astype(int)

        # Assume images well centered, so pick largest area object if >1 per image
        area = (arr[:, 3] - arr[:, 1]) * (arr[:, 2] - arr[:, 0])  # Format: xyxy
        if (area < min_area).all():  # if none adequately sized
            continue
        else:
            arr = arr[np.argmax(area)]

        # Crop to bounding box of vehicle
        cropped = img[arr[1]:arr[3], arr[0]:arr[2]]

        # Resize
        cropped = cv2.resize(cropped, (resize, resize), interpolation=cv2.INTER_LINEAR)

        # Image name
        img_name = '_'.join(foo['Path'].split('/')[-4:-1]) + '_' + str(x) + '.png'

        # Output image
        cv2.imwrite(os.path.join(outputDir, img_name), cropped)

if __name__ == '__main__':

    # Read data
    rootDir = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier'
    df = pd.read_csv(os.path.join(rootDir, 'Vehicle Make Model Directory Bboxes.csv'))

    # Convert str representation of bbox list to list
    df['Bboxes'] = df['Bboxes'].apply(lambda x: list(ast.literal_eval(x)))

    # Restrict to rows with bounding boxes
    df = df.loc[df.Bboxes.str.len() != 0].reset_index(drop=True)

    # Number of vehicles
    df['Nr Veh'] = df['Bboxes'].apply(lambda x: len(x))

    # Output directory
    outputDir = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier/cropped'

    main(df, outputDir, resize=400, min_area=50625, min_confidence=0.5)