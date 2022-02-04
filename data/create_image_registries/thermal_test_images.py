import pandas as pd
import os
import cv2
import numpy as np
import argparse

pd.set_option('display.max_columns', 500)

"""
    Script generates image registries (2) for thermal and visible image pairs. First, split paired images into
    `visible` and `thermal` subdirectories, retaining same filename per image. Second, create image an image
    registry for both subdirectories. Precursor script to `run_yolov5.py` for these data.
    
    Input data are paired thermal (left) visible (right) images from the MERGEN parking lot series. These
    were hand labeled with the vehicle make-model class, in the filename itself.
"""

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='Path to root directory of images', required=True)
    parser.add_argument('--output', type=str, help='Output path, to include title of CSV file', required=True)
    args = parser.parse_args()
    return args

def split_images(data_path):
    """
    Splits matched visible and thermal images into separate images in subdirectories
    :param data_path: str, path to root dir of images
    :returns: None
    """
    files = sorted([i for i in os.listdir(data_path) if "png" in i or "jpg" in i])

    df = pd.Series(files)

    os.makedirs(os.path.join(data_path, 'visible'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'thermal'), exist_ok=True)

    for i in range(len(df)):

        if i % 100 == 0:
            print(i)

        img = cv2.imread(os.path.join(data_path, df.iloc[i]))

        half = img.shape[1] // 2

        # split into visible and thermal
        visible = img[:img.shape[0], :half]
        thermal = img[:img.shape[0], half:]

        # output visible and thermal
        cv2.imwrite(os.path.join(data_path, 'visible', df.iloc[i]), visible)
        cv2.imwrite(os.path.join(data_path, 'thermal', df.iloc[i]), thermal)

def main(opt):

    # Split images
    split_images(opt.root)

    for subdir in ['visible', 'thermal']:

        path = os.path.join(opt.root, subdir)

        files = [subdir+'/'+i for i in os.listdir(path) if 'png' in i or 'jpg' in i]

        df = pd.DataFrame(files, columns=['Source Path'])

        # Clean make-model
        df['Make-Model'] = df['Source Path'].apply(lambda x: ''.join(x.split('/')[-1]))
        df['Make-Model'] = df['Make-Model'].str[:-4]  # removes file suffix
        df['Make-Model'] = df['Make-Model'].apply(lambda x: ' '.join(x.split('-')[1:-1]))

        df['Make-Model'] = np.where(df['Make-Model'] == 'FORD TAURUS', 'Ford Taurus', df['Make-Model'])
        df['Make-Model'] = np.where(df['Make-Model'] == 'ford f150', 'Ford F-Series', df['Make-Model'])
        df['Make-Model'] = np.where(df['Make-Model'] == 'TOYOTA PRIUS', 'Toyota Prius', df['Make-Model'])
        df['Make-Model'] = np.where(df['Make-Model'] == 'TOYOTA CAMRY', 'Toyota Camry', df['Make-Model'])
        df['Make-Model'] = np.where(df['Make-Model'] == 'TOYOTA TUNDRA', 'Toyota Tundra', df['Make-Model'])
        df['Make-Model'] = np.where(df['Make-Model'] == 'HONDA CRV', 'Honda CR-V', df['Make-Model'])
        df['Make-Model'] = np.where(df['Make-Model'] == 'JEEP WRANGLER', 'Jeep Wrangler', df['Make-Model'])
        df['Make-Model'] = np.where(df['Make-Model'] == 'JEEP CHEROKEE', 'Jeep Cherokee', df['Make-Model'])
        df['Make-Model'] = np.where(df['Make-Model'] == 'HYUNDAI SANTA FE', 'Hyundai Santa Fe', df['Make-Model'])
        df['Make-Model'] = np.where(df['Make-Model'] == 'VOLVO XC90', 'Volvo XC90', df['Make-Model'])
        df['Make-Model'] = np.where(df['Make-Model'] == 'MERCEDES C CLASS', 'Mercedes-Benz C-Class', df['Make-Model'])

        df['Make'] = df['Make-Model'].apply(lambda x: ''.join(x.split(' ')[0]))
        df['Model'] = df['Make-Model'].apply(lambda x: ''.join(x.split(' ')[1:]))

        df = df[['Make', 'Model', 'Source Path']]

        df = df.sort_values(by=['Make', 'Model', 'Source Path'])

        output = opt.output[:-4] + f'_{subdir}' + '.csv'

        df.to_csv(output, index=False)


if __name__ == '__main__':

    opt = parse_opt()
    main(opt)