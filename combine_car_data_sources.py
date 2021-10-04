import pandas as pd
import os
import re
import numpy as np
from shutil import copy, rmtree

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', None)

def create_paths(rootDir):
    '''
    Creates pd.DataFrame of files and full paths
    :param rootDir: str, path to root directory
    :return: pd.DataFrame
    '''
    # Create pd.DataFrame of vehicle make models based on directory structure
    lst = []
    for subdir, dirs, files in os.walk(rootDir):
        for file in [i for i in files if 'jpg' in i or 'png' in i]:
            lst.append('/'.join(os.path.join(subdir, file).split('/')[-4:]))
    if rootDir.split('/')[-1] == 'stanford_car_data':
        lst = ['/'.join(rootDir.split('/')[:-1])+'/'+i for i in lst]  # make absolute path
    else:
        lst = ['/'.join(rootDir.split('/')[:-2]) + '/' + i for i in lst]  # dir structure is different
    return pd.DataFrame(lst, columns=['Orig Path'])


if __name__ == '__main__':

    output_path = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier/combined_car_data'
    if os.path.exists(output_path):  # delete dir if exists
        rmtree(output_path)

    #####################################################
    #####           Stanford Car Dataset            #####
    #####################################################

    stanford_data = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier/stanford_car_data'

    df = create_paths(stanford_data)

    # Create columns
    df['Make'] = df['Orig Path'].apply(lambda x: x.split('/')[-2].split(' ')[0])
    df['Make'] = np.where(df['Make'] == 'Aston', 'Aston Martin', df['Make'])
    df['Make'] = np.where(df['Make'] == 'AM', 'AM General', df['Make'])
    df['Model'] = df['Orig Path'].apply(lambda x: ' '.join(x.split('/')[-2].split(' ')[1:-1]))
    df['Model'] = df['Model'].str.replace('Martin', '').str.replace('General Hummer', 'Hummer')
    df['Year'] = df['Orig Path'].apply(lambda x: x.split('/')[-2].split(' ')[-1]).astype(int)
    for col in ['Make', 'Model']:
        df[col] = df[col].str.strip()  # Remove whitespace, if any

    df = df.sort_values(by=['Make', 'Model', 'Year']).reset_index(drop=True)
    df['Model'] = df['Model'].str.lower()

    lst = []
    # Move files and rename
    for i in range(len(df['Orig Path'])):
        output = os.path.join(output_path, df['Make'].iloc[i], df['Model'].iloc[i], str(df['Year'].iloc[i]))
        os.makedirs(output, exist_ok=True)
        copy(df['Orig Path'].iloc[i], output)  # copy file to new dest with orig name
        old = os.path.join(output, df['Orig Path'].iloc[i].split('/')[-1])  # old name
        new = os.path.join(output, 'stanford_'+ str(i) + '.' + df['Orig Path'].iloc[i].split('/')[-1].split('.')[-1])
        os.rename(old, new)
        lst.append(new)

    df = pd.concat([df, pd.DataFrame(lst, columns=['New Path'])], axis=1)

    df = df[['Make', 'Model', 'Year', 'Orig Path', 'New Path']]

    df.to_csv(os.path.join(stanford_data, '../', 'stanford_car_directory.csv'), index=False)
    del df, lst

    #####################################################
    #####                   VMMRdb                  #####
    #####################################################

    vmmr = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier/VMMRdb'

    df = create_paths(vmmr)

    # Note - columns standardized to Stanford dataset

    # Make
    df['Make'] = df['Orig Path'].apply(lambda x: x.split('/')[-2].split(' ')[0])
    df['Make'] = df['Make'].apply(lambda x: x.split('_')[0] if len(x.split('_')) else x)
    df['Make'] = np.where(df['Make'] == 'mercedes', 'Mercedes-Benz', df['Make'])
    df['Make'] = np.where(df['Make'] == 'aston', 'Aston Martin', df['Make'])
    df['Make'] = np.where(df['Make'] == 'can', 'Can-Am', df['Make'])
    df['Make'] = np.where(df['Make'] == 'am', 'AM General', df['Make'])
    df['Make'] = np.where(df['Make'] == 'alfa', 'Alfa Romeo', df['Make'])
    df['Make'] = np.where(df['Make'] == 'rollsroyce', 'Rolls-Royce', df['Make'])
    fixed = ['Mercedes-Benz', 'Aston Martin', 'Can-Am', 'AM General', 'Alfa Romeo', 'Rolls-Royce']

    # All caps
    allcaps = ['ram', 'bmw', 'amc', 'mg', 'mini', 'gmc', 'fiat']
    for x in allcaps:
        df['Make'] = np.where(df['Make'] == x, x.upper(), df['Make'])

    # Upper case first letter
    remainders = [i for i in df.Make.unique().tolist() if i not in fixed+[i.upper() for i in allcaps]]
    for x in remainders:
        df['Make'] = np.where(df['Make'] == x, x.capitalize(), df['Make'])

    # Year
    df['Year'] = df['Orig Path'].apply(lambda x: x.split('/')[-2][-4:]).astype(int)
    df['Year'] = np.where(df.Year == 1900, 1990, df['Year'])  # Mistake in coding 1990 as 1900
    df['Year'] = np.where(df.Year == 1908, 1998, df['Year'])  # Mistake

    # Model
    foo = df['Orig Path'].apply(lambda x: x.split('/')[-2][:-5])
    foo = foo.apply(lambda x: ' '.join(re.split(r'\s|_', x, maxsplit=0)))
    make_unformatted = df['Make'].apply(lambda x: ' '.join(x.split('-')).lower())

    lst = []
    for i in foo.index:
        tmp = ' '.join([j for j in foo[i].split() if j not in make_unformatted[i].split()])
        lst.append(tmp)
    df['Model'] = pd.Series(lst)

    # Move files to new directories
    lst = []
    for i in range(len(df)):
        output = os.path.join(output_path, df['Make'].iloc[i], df['Model'].iloc[i], str(df['Year'].iloc[i]))
        os.makedirs(output, exist_ok=True)  # will just add if already exits
        copy(df['Orig Path'].iloc[i], output)  # copy file to new dest with orig name
        old = os.path.join(output, df['Orig Path'].iloc[i].split('/')[-1])  # old name
        new = os.path.join(output, 'vmmr' + '_' + str(i) + '.' + df['Orig Path'].iloc[i].split('/')[-1].split('.')[-1])
        os.rename(old, new)
        lst.append(new)

    df = pd.concat([df, pd.DataFrame(lst, columns=['New Path'])], axis=1)

    df = df[['Make', 'Model', 'Year', 'Orig Path', 'New Path']]

    df.to_csv(os.path.join(stanford_data, '../', 'vmmr_car_directory.csv'), index=False)