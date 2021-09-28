import pandas as pd
import cv2
import os
import ast
import numpy as np
import secrets

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', None)


def main(df, outputDir, resize=400, min_veh_pixels=150, min_confidence=0.5):
    """
    Loops through all image paths in dataframe `df`, restricts to bounding box coordinates in `df['Bboxes']`,
        resizes image and outputs to directory.
    :param df: pd.DataFrame
    :param outputDir: str, path to output each image
    :param resize: int, w x h pixel dimensions of final image
    :param min_veh_pixels: int, minimimum w x h pixel dimensions for a vehicle to be output
    :param min_confidence: float, minimum confidence threshold of vehicle output
    :return:
    """

    for x in range(len(df)):

        # Read in single image
        foo = df.iloc[x]
        img = cv2.imread(foo['Path'])

        # Crop separately for each vehicle in image
        for i in range(foo['Nr Veh']):

            # Skip vehicle if below minimum confidence threshold
            if foo['Bboxes'][i][4] < min_confidence:
                continue

            bbox = np.array([int(j) for j in foo['Bboxes'][i][:4]]) # xyxy format, 5th element is confidence level

            # Skip if number of pixels in car image is really small
            if (bbox[3]-bbox[1] < min_veh_pixels) or (bbox[2]-bbox[0] < min_veh_pixels):
                continue

            # Crop to bounding box of vehicle
            cropped = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            # Resize
            cropped = cv2.resize(cropped, (resize, resize), interpolation=cv2.INTER_LINEAR)

            # Generate random hash
            hex = secrets.token_hex(nbytes=2)
            img_name = '_'.join(foo['Path'].split('/')[-4:-1])

            # Output
            cv2.imwrite(os.path.join(outputDir, img_name+'_'+hex+'.png'), cropped)

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

    main(df, outputDir, resize=400, min_veh_pixels=150, min_confidence=0.5)