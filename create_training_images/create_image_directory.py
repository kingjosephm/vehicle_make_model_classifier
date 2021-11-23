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

    # Fixes to account for Chevrolet C/K and RAM C/V
    df.loc[(df.Make == 'Chevrolet') & (df.Model == 'C:K'), 'Model'] = 'C/K'  # Python changes `/` to `:`
    df.loc[(df.Make == 'RAM') & (df.Model == 'C:V'), 'Model'] = 'C/V'

    ################################
    ##### Source URL per Image #####
    ################################

    path = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier/data_directories/_image_sources.json'

    with open(path, 'rb') as j:
        url = json.load(j)

    urls = pd.DataFrame(url.items(), columns=['Source Path', 'URL'])

    ##########################################
    ##### Append source URL to DataFrame #####
    ##########################################

    df = df.merge(urls, on='Source Path', how='left')

    ######################################
    ##### Clean URL, drop duplicates #####
    ######################################

    df['valid_url'] = df['URL'].apply(lambda x: valid_URL(x))
    df['URL'] = np.where(df['valid_url'] == False, np.NaN, df['URL'])

    missings = df.loc[df.URL.isnull()]

    df = df.loc[df.URL.notnull()]

    df['dup'] = df.duplicated(subset=["URL"])
    dups = df.loc[df.dup == True].reset_index(drop=True)
    df = df.loc[df.dup == False].reset_index(drop=True)

    df = df[['Make', 'Model', 'Source Path', 'URL']]

    ########################################
    ##### Delete duplicates from disk ######
    ########################################

    for x in dups['Source Path']:
        os.remove(x)

    for x in missings['Source Path']:
        os.remove(x)

    ##############################################
    ##### Output corrected image source JSON #####
    ##############################################

    img_source = df[['Source Path', 'URL']].set_index("Source Path").to_dict(orient='index')
    for key, val in img_source.items():
        img_source[key] = val['URL']

    with open(path, 'w') as j:
        json.dump(img_source, j)

    #####################################
    ##### Merge vehicle type column #####
    #####################################

    db = pd.read_csv('../data/make_model_database_mod.csv')
    db = db[['Make', 'Model', 'Category']].drop_duplicates(subset=['Make', 'Model', 'Category']).reset_index(drop=True)
    db['Category'] = np.where(db.Category.isin(['Coupe', 'Sedan', 'Hatchback', 'Convertible', 'Wagon']), 'Car', db['Category'])
    db = db.drop_duplicates().reset_index(drop=True)

    df = df.merge(db, on=['Make', 'Model'])

    df = df.sort_values(by=['Make', 'Model'])

    df = df[['Make', 'Model', 'Category', 'Source Path', 'URL']]

    outputPath = './data'
    os.makedirs(outputPath, exist_ok=True)
    df.to_csv(os.path.join(outputPath, 'MakeModelDirectory.csv'), index=False)