import pandas as pd
import os
import json
import numpy as np
import validators
import argparse

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', None)

"""
    Script creates image registry for scraped Google images (training set). This is the precursor script before running
    `run_yolov5.py`. This script creates pd.DataFrame with relative image paths, make-model labels, vehicle category,
    and image source URL. Labels are derived from directory structure of images themselves.
"""

def valid_URL(url):
    try:
        return True if validators.url(url) else False
    except TypeError:
        return False

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='Path to root directory of images', required=True)
    parser.add_argument('--image-sources', type=str, help='Full path to `image_sources.json` for scraped images', required=True)
    parser.add_argument('--make-model-db', type=str, help='Full path to `make_model_database_mod.csv` for vehicle category', required=True)
    parser.add_argument('--output', type=str, help='Output path, to include title of CSV file', required=True)
    args = parser.parse_args()
    return args

def main(opt):

    ###############################
    ##### DataFrame of Images #####
    ###############################

    lst = []
    for subdir, dirs, files in os.walk(opt.root):
        for file in [i for i in files if 'jpg' in i or 'png' in i]:
            lst.append('/'.join(os.path.join(subdir, file).split('/')[-4:]))  # does not count empty subdirectories
    df = pd.DataFrame(lst, columns=["Source Path"])
    df['Make'] = df['Source Path'].apply(lambda x: x.split('/')[0])
    df['Model'] = df['Source Path'].apply(lambda x: x.split('/')[1])
    df['Year'] = df['Source Path'].apply(lambda x: x.split('/')[2]).astype(int)

    # Fixes to account for Chevrolet C/K and RAM C/V
    df.loc[(df.Make == 'Chevrolet') & (df.Model == 'C:K'), 'Model'] = 'C/K'  # Python changes `/` to `:`
    df.loc[(df.Make == 'RAM') & (df.Model == 'C:V'), 'Model'] = 'C/V'

    #####################################
    ##### Ensure Image URLs unique  #####
    #####################################

    with open(opt.image_sources, 'rb') as j:
        url = json.load(j)

    urls = pd.DataFrame(url.items(), columns=['Source Path', 'URL'])

    df = df.merge(urls, on='Source Path', how='left')

    # Verify all images have URLs
    assert(df['URL'].isnull().mean() == 0), "Images missing source URLs!"  # verify all images have URL

    # Verify URLs are valid (in order to verify origin)
    valid_url = df['URL'].apply(lambda x: valid_URL(x))
    assert(valid_url.mean() == 1), "Invalid URLs found!"

    # Verify image URLs unique as proxy for image unduplication
    assert(df.duplicated(subset=['URL']).mean() == 0), "images with duplicate URLs found!"

    df = df[['Make', 'Model', 'Source Path', 'URL']]

    orig_len = len(df)

    db = pd.read_csv(opt.make_model_db)
    db = db[['Make', 'Model', 'Category']].drop_duplicates().reset_index(drop=True)
    db['Category'] = np.where(db.Category.isin(['Coupe', 'Sedan', 'Hatchback', 'Convertible', 'Wagon']), 'Car', db['Category'])

    # Get modal category, since some change over time
    db = db.groupby(['Make', 'Model'])['Category'].agg(lambda x: x.value_counts().index[0]).reset_index()

    df = df.merge(db, on=['Make', 'Model'])

    assert (orig_len == len(df)), 'Length mismatch post-merge!'
    assert (df['Category'].isnull().mean() == 0), "Missing vehicle category post-merge!"

    df = df.sort_values(by=['Make', 'Model'])

    df = df[['Make', 'Model', 'Category', 'Source Path', 'URL']].reset_index(drop=True)

    df.to_csv(opt.output, index=False)

if __name__ == '__main__':

    opt = parse_opt()
    main(opt)