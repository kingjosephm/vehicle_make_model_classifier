import pandas as pd
import os
import json
import numpy as np
import validators

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', None)

def valid_URL(url):
    try:
        return True if validators.url(url) else False
    except TypeError:
        return False

if __name__ == '__main__':


    rootDir = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier/scraped_images'

    number_images = 100

    ###############################
    ##### DataFrame of Images #####
    ###############################

    lst = []
    for subdir, dirs, files in os.walk(rootDir):
        for file in [i for i in files if 'jpg' in i or 'png' in i]:
            lst.append('/'.join(os.path.join(subdir, file).split('/')[-4:]))  # does not count empty subdirectories
    df = pd.DataFrame(lst, columns=["Source Path"])
    df['Make'] = df['Source Path'].apply(lambda x: x.split('/')[0])
    df['Model'] = df['Source Path'].apply(lambda x: x.split('/')[1])
    df['Year'] = df['Source Path'].apply(lambda x: x.split('/')[2]).astype(int)
    df['dir'] = df['Source Path'].apply(lambda x: '/'.join(x.split('/')[:-1]))

    # Fixes to account for Chevrolet C/K and RAM C/V
    df.loc[(df.Make == 'Chevrolet') & (df.Model == 'C:K'), 'Model'] = 'C/K'  # Python changes `/` to `:`
    df.loc[(df.Make == 'RAM') & (df.Model == 'C:V'), 'Model'] = 'C/V'

    # Remove directories still scraping images - incomplete
    df['image_count'] = df.groupby(['Make', 'Model', 'Year'])['Source Path'].transform('count')
    incomplete_dirs = df.loc[df['image_count'] < number_images-30]['dir'].drop_duplicates().tolist()
    if incomplete_dirs:
        for x in incomplete_dirs:
            df = df.loc[df['dir'] != x].reset_index(drop=True)

    df['Source Path'] = df['Source Path'].apply(lambda x: rootDir + '/' + x)  # Makes absolute path
    del df['image_count']

    ####################################
    ##### DataFrame of JSON files ######
    ####################################

    lst2 = []
    for subdir, dirs, files in os.walk(rootDir):
        for file in [i for i in files if 'json' in i]:
            lst2.append('/'.join(os.path.join(subdir, file).split('/')[-4:]))  # does not count empty subdirectories
    foo = pd.DataFrame(lst2, columns=["Path"])
    foo['Make'] = foo['Path'].apply(lambda x: x.split('/')[0])
    foo['Model'] = foo['Path'].apply(lambda x: x.split('/')[1])
    foo['Year'] = foo['Path'].apply(lambda x: x.split('/')[2]).astype(int)
    foo['dir'] = foo['Path'].apply(lambda x: '/'.join(x.split('/')[:-1]))

    # Fixes to account for Chevrolet C/K and RAM C/V
    foo.loc[(foo.Make == 'Chevrolet') & (foo.Model == 'C:K'), 'Model'] = 'C/K'  # Python changes `/` to `:`
    foo.loc[(foo.Make == 'RAM') & (foo.Model == 'C:V'), 'Model'] = 'C/V'

    foo['Path'] = foo['Path'].apply(lambda x: rootDir + '/' + x)  # Makes absolute path

    ##########################################
    ##### Append source URL to DataFrame #####
    ##########################################

    temp_df = df[['dir']].drop_duplicates().sort_values(by='dir').reset_index(drop=True)  # de-duplicated list of directories with sufficient number of images
    foo = foo.merge(temp_df, on=['dir'], how='inner')  # merge restricts to JSON files in directories with sufficient number of images, otherwise includes dirs still being scraped
    foo = foo.sort_values(by=['Make', 'Model', 'Year']).reset_index(drop=True)
    del df['dir']  # no longer needed

    url_list = []

    for i in foo.index:

        with open(foo.iloc[i, 0], 'rb') as j:
            temp = json.load(j)
            for key, value in temp.items():
                url_list.append(['/'.join(foo.iloc[i, 0].split('/')[:-1]) + '/' + key, value])

    urls = pd.DataFrame(url_list, columns=['Source Path', 'URL'])

    df = df.merge(urls, on='Source Path', how='left')

    ######################################
    ##### Clean URL, drop duplicates #####
    ######################################

    df['valid_url'] = df['URL'].apply(lambda x: valid_URL(x))
    df['URL'] = np.where(df['valid_url'] == False, np.NaN, df['URL'])

    df = df.loc[df.URL.notnull()]

    df = df.drop_duplicates(subset=['URL'])

    df = df[['Make', 'Model', 'Source Path', 'URL']]

    #####################################
    ##### Merge vehicle type column #####
    #####################################

    db = pd.read_csv('./create_training_images/make_model_database_mod.csv')
    db = db[['Make', 'Model', 'Category']].drop_duplicates(subset=['Make', 'Model', 'Category']).reset_index(drop=True)
    db['Category'] = np.where(db.Category.isin(['Coupe', 'Sedan', 'Hatchback', 'Convertible', 'Wagon']), 'Car', db['Category'])
    db = db.drop_duplicates().reset_index(drop=True)

    df = df.merge(db, on=['Make', 'Model'])

    df = df.sort_values(by=['Make', 'Model'])

    df = df[['Make', 'Model', 'Category', 'Source Path', 'URL']]

    outputPath = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier/data_directories'
    df.to_csv(os.path.join(outputPath, 'MakeModelDirectory.csv'), index=False)