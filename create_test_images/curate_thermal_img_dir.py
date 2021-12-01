import pandas as pd
import os
import cv2
import numpy as np
pd.set_option('display.max_columns', 500)

def split_images(data_path):
    """
    Splits matched visible and thermal images into separate images in subdirectories
    :param data_path: str, path to root dir of images
    :returns: None
    """
    files = sorted([i for i in os.listdir(data_path) if "png" in i or "jpg" in i])

    df = pd.Series(files)

    for i in range(len(df)):

        if i % 100 == 0:
            print(i)

        img = cv2.imread(os.path.join(data_path, df.iloc[i]))

        half = img.shape[1] // 2

        # split into visible and thermal
        visible = img[:img.shape[0], :half]
        thermal = img[:img.shape[0], half:]

        # output visible and thermal
        cv2.imwrite(os.path.join(data_path, '../test_/visible', df.iloc[i]), visible)
        cv2.imwrite(os.path.join(data_path, '..test_/thermal', df.iloc[i]), thermal)

def create_image_df(rootDir):

    lst = []
    for subdir, dirs, files in os.walk(rootDir):
        for file in [i for i in files if 'jpg' in i or 'png' in i]:
            lst.append('/'.join(os.path.join(subdir, file).split('/')[-2:]))

    df = pd.DataFrame(lst, columns=['Source Path'])
    df['visible'] = df['Source Path'].apply(lambda x: ''.join(x.split('/')[0]))
    df['visible'] = np.where(df['visible'] == 'visible', True, False)
    assert (len(df[df.visible == True]) == len(df[df.visible == False]))

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

    df = df[['Make', 'Model', 'Source Path', 'visible']]

    df = df.sort_values(by=['Source Path', 'visible'])

    df.to_csv(os.path.join(rootDir, 'MakeModelDir.csv'), index=False)


if __name__ == '__main__':

    data_path = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/pix2pix/MERGEN/test'
    split_images(data_path)

    split_img_path = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/pix2pix/MERGEN/test_'
    create_image_df(split_img_path)



