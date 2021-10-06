import pandas as pd
import os
import re
import json
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

    with open('utils/stanford_model_fixes.json') as f:
        stanford_model_fixes = json.load(f)

    stanford_data = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier/stanford_car_data'

    df = create_paths(stanford_data)

    # Make
    df['Make'] = df['Orig Path'].apply(lambda x: x.split('/')[-2].split(' ')[0])
    df['Make'] = np.where(df['Make'] == 'Aston', 'Aston Martin', df['Make'])
    df['Make'] = np.where(df['Make'] == 'AM', 'AM General', df['Make'])
    df['Make'] = np.where(df['Make'] == 'Land', 'Land Rover', df['Make'])
    df['Make'] = np.where(df["Make"] == 'Ram', 'RAM', df['Make'])

    # Model
    df['Model'] = df['Orig Path'].apply(lambda x: ' '.join(x.split('/')[-2].split(' ')[1:-1]))
    df['Model'] = df['Model'].str.replace('Martin', '').str.replace('General Hummer', 'Hummer')
    df['Model'] = df['Model'].str.replace('Rover LR2', 'LR2').str.replace('Rover Range Rover', 'Range Rover')
    df['Model'] = df['Model'].str.lower()  # Easier to standardize with vmmr dataset

    # Remove whitespace
    for col in ['Make', 'Model']:
        df[col] = df[col].apply(lambda x: ' '.join([i for i in x.split()]))

    for key, val in stanford_model_fixes.items():
        df['Model'] = df['Model'].str.replace(rf'\b{key}\b', val, regex=True)
        df['Model'] = df['Model'].apply(lambda x: ' '.join(x.split())) # remove whitespace, if above created any

    # Fix multiple versions of Audi tt
    df['Model'] = np.where((df.Make == 'Audi') & (df['Model'] == 'tt rs'), 'tt', df['Model'])
    df['Model'] = np.where((df.Make == 'Audi') & (df['Model'] == 'tts'), 'tt', df['Model'])

    df['Model'] = np.where(df.Model=='continental supersports conv.', 'continental', df['Model'])

    make_models_stanford = df[['Make', 'Model']].drop_duplicates(subset=["Make", "Model"]).sort_values(by=['Make', 'Model']).reset_index(drop=True)

    # Year
    df['Year'] = df['Orig Path'].apply(lambda x: x.split('/')[-2].split(' ')[-1]).astype(int)

    df = df.sort_values(by=['Make', 'Model', 'Year']).reset_index(drop=True)

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

    #####################################################
    #####                   VMMRdb                  #####
    #####################################################

    vmmr = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier/VMMRdb'

    with open('utils/vmmr_model_fixes.json') as f:
        vmmr_model_fixes = json.load(f)

    df2 = create_paths(vmmr)

    # Note - columns standardized to Stanford dataset

    # Make
    df2['Make'] = df2['Orig Path'].apply(lambda x: x.split('/')[-2].split(' ')[0])
    df2['Make'] = df2['Make'].apply(lambda x: x.split('_')[0] if len(x.split('_')) else x)
    df2['Make'] = np.where(df2['Make'] == 'mercedes', 'Mercedes-Benz', df2['Make'])
    df2['Make'] = np.where(df2['Make'] == 'aston', 'Aston Martin', df2['Make'])
    df2['Make'] = np.where(df2['Make'] == 'can', 'Can-Am', df2['Make'])
    df2['Make'] = np.where(df2['Make'] == 'am', 'AM General', df2['Make'])
    df2['Make'] = np.where(df2['Make'] == 'alfa', 'Alfa Romeo', df2['Make'])
    df2['Make'] = np.where(df2['Make'] == 'rollsroyce', 'Rolls-Royce', df2['Make'])
    fixed = ['Mercedes-Benz', 'Aston Martin', 'Can-Am', 'AM General', 'Alfa Romeo', 'Rolls-Royce']

    allcaps = ['ram', 'bmw', 'amc', 'mg', 'mini', 'gmc', 'fiat']
    for x in allcaps:
        df2['Make'] = np.where(df2['Make'] == x, x.upper(), df2['Make'])

    remainders = [i for i in df2.Make.unique().tolist() if i not in fixed+[i.upper() for i in allcaps]]
    for x in remainders:
        df2['Make'] = np.where(df2['Make'] == x, x.capitalize(), df2['Make'])

    df2['Make'] = df2.Make.str.replace('Landrover', "Land Rover")

    # Year
    df2['Year'] = df2['Orig Path'].apply(lambda x: x.split('/')[-2][-4:]).astype(int)
    df2['Year'] = np.where(df2.Year == 1900, 1990, df2['Year'])  # Mistake in coding 1990 as 1900
    df2['Year'] = np.where(df2.Year == 1908, 1998, df2['Year'])  # Mistake

    # Model
    foo = df2['Orig Path'].apply(lambda x: x.split('/')[-2][:-5])
    foo = foo.apply(lambda x: ' '.join(re.split(r'\s|_', x, maxsplit=0)))
    make_unformatted = df2['Make'].apply(lambda x: ' '.join(x.split('-')).lower())

    lst = []
    for i in foo.index:
        tmp = ' '.join([j for j in foo[i].split() if j not in make_unformatted[i].split()])
        lst.append(tmp)
    df2['Model'] = pd.Series(lst)

    for col in ['Make', 'Model']:
        df2[col] = df2[col].str.strip()  # Remove whitespace, if any

    for key, val in vmmr_model_fixes.items():
        df2['Model'] = df2['Model'].str.replace(rf'\b{key}\b', val, regex=True)
        df2['Model'] = df2['Model'].apply(lambda x: ' '.join(x.split())) # remove whitespace, if above created any

    # GMC trucks
    for x in ['c1000', 'c1500', 'c2500', 'c3500', 'c4500']:
        df2['Model'] = np.where((df2.Make == 'GMC') & (df2.Model == x), 'c-series', df2['Model'])
    for x in ['c5500', 'c7500']:
        df2['Model'] = np.where((df2.Make == 'GMC') & (df2.Model == x), 'c6500', df2['Model'])
    for x in ['k10', 'k20', 'k15', 'k1500', 'k2500', 'k3500']:
        df2['Model'] = np.where((df2.Make == 'GMC') & (df2.Model == x), 'k-series', df2['Model'])

    # Ford trucks
    for x in ['f100', 'f150', 'f250', 'f350', 'f450']:
        df2['Model'] = np.where((df2.Make == 'Ford') & (df2.Model == x), 'f-series', df2['Model'])
    for x in ['w100', 'e250', 'e350', 'e450']:
        df2['Model'] = np.where((df2.Make == 'Ford') & (df2.Model == x), 'e-series', df2['Model'])

    # Dodge trucks
    for x in ['w100', 'w150', 'w200', 'w250', 'w350', 'd100', 'd150', 'd200', 'd250', 'd350']:  # identical except `w` is 4-wheel drive, `d` is 2
        df2['Model'] = np.where((df2.Make == 'Dodge') & (df2.Model == x), 'w-series', df2['Model'])
    for x in ['ram 1500', 'ram 2500', 'ram 3500', 'ram 4500', 'ram 5500']:
        df2['Model'] = np.where((df2.Make == 'Dodge') & (df2.Model == x), 'ram', df2['Model'])

    # Chevy trucks
    for x in ['v10', 'v20', 'k10', 'k20', 'c10', 'c20', 'c30', 'c-k1500', 'c-k2500', 'c-k3500']:  # again, 2- vs 4-wheel drive
        df2['Model'] = np.where((df2.Make == 'Chevrolet') & (df2.Model == x), 'ck-series', df2['Model'])
    for x in ['g10', 'g20', 'g30']:
        df2['Model'] = np.where((df2.Make == 'Chevrolet') & (df2.Model == x), 'g-series', df2['Model'])

    # Hummer mislabeled
    df2['Make'] = np.where((df2.Make == "Hummer") & (df2.Model == 'h1'), 'AM General', df2['Make'])
    df2['Model'] = np.where((df2.Make == "AM General") & (df2.Model == 'h1'), 'hummer', df2['Model'])

    # Mazda truck
    for x in ['b2000', 'b2200', 'b2300', 'b2500', 'b2600', 'b3000', 'b4000']:
        df2['Model'] = np.where((df2.Make == 'Mazda') & (df2.Model == x), 'b-series', df2['Model'])

    # Fix Tesla
    df2['Model'] = np.where((df2['Make'] == 'Tesla') & (df2['Model'] == 's'), 'model s', df2['Model'])

    # Combine RAM pickups
    for x in ['1500', '2500', '3500']:
        df2['Model'] = np.where((df2.Make == 'RAM') & (df2.Model == x), 'pickup', df2['Model'])

    # BMW
    for x in ['325ci', '325e', '325es', '325i', '325is', '325xi']:
        df2['Model'] = np.where((df2.Make == 'BMW') & (df2.Model == x), '325', df2['Model'])
    for x in ['328i', '328ic', '328is', '328xi']:
        df2['Model'] = np.where((df2.Make == 'BMW') & (df2.Model == x), '328', df2['Model'])
    for x in ['330ci', '330i', '330xi']:
        df2['Model'] = np.where((df2.Make == 'BMW') & (df2.Model == x), '330', df2['Model'])
    for x in ['335i', '335xi']:
        df2['Model'] = np.where((df2.Make == 'BMW') & (df2.Model == x), '335', df2['Model'])
    for x in ['525i', '525xi']:
        df2['Model'] = np.where((df2.Make == 'BMW') & (df2.Model == x), '525', df2['Model'])
    for x in ['528e', '528i', '528xi']:
        df2['Model'] = np.where((df2.Make == 'BMW') & (df2.Model == x), '528', df2['Model'])
    for x in ['530i', '530xi']:
        df2['Model'] = np.where((df2.Make == 'BMW') & (df2.Model == x), '530', df2['Model'])
    for x in ['535i', '535xi']:
        df2['Model'] = np.where((df2.Make == 'BMW') & (df2.Model == x), '535', df2['Model'])
    for x in ['645ci', '645i']:
        df2['Model'] = np.where((df2.Make == 'BMW') & (df2.Model == x), '645', df2['Model'])
    for x in ['840ci', '850ci', '850i']:
        df2['Model'] = np.where((df2.Make == 'BMW') & (df2.Model == x), '850', df2['Model'])

    # Mercedes
    df2['Model'] = np.where((df2.Make == 'Mercedes-Benz') & (df2.Model == 'e190'), '190', df2['Model'])
    df2['Model'] = np.where((df2.Make == 'Mercedes-Benz') & (df2.Model == 'ce300'), '300ce', df2['Model'])
    df2['Model'] = np.where((df2.Make == 'Mercedes-Benz') & (df2.Model == 'cd300'), '300ce', df2['Model'])
    df2['Model'] = np.where((df2.Make == 'Mercedes-Benz') & (df2.Model == 'sd300'), '300sd', df2['Model'])
    df2['Model'] = np.where((df2.Make == 'Mercedes-Benz') & (df2.Model == 'sdl300'), '300sd', df2['Model'])
    df2['Model'] = np.where((df2.Make == 'Mercedes-Benz') & (df2.Model == 'se380'), '380se', df2['Model'])
    df2['Model'] = np.where((df2.Make == 'Mercedes-Benz') & (df2.Model == 'se400'), '400se', df2['Model'])
    df2['Model'] = np.where((df2.Make == 'Mercedes-Benz') & (df2.Model == 'sel400'), '400se', df2['Model'])
    df2['Model'] = np.where((df2.Make == 'Mercedes-Benz') & (df2.Model == 'se450'), '450se', df2['Model'])
    df2['Model'] = np.where((df2.Make == 'Mercedes-Benz') & (df2.Model == 'sel450'), '450se', df2['Model'])
    df2['Model'] = np.where((df2.Make == 'Mercedes-Benz') & (df2.Model == 'sel300'), '300sel', df2['Model'])
    df2['Model'] = np.where((df2.Make == 'Mercedes-Benz') & (df2.Model == 'sel400'), '400sel', df2['Model'])
    df2['Model'] = np.where((df2.Make == 'Mercedes-Benz') & (df2.Model == 'sel420'), '420sel', df2['Model'])
    df2['Model'] = np.where((df2.Make == 'Mercedes-Benz') & (df2.Model == 'sel450'), '450sel', df2['Model'])
    df2['Model'] = np.where((df2.Make == 'Mercedes-Benz') & (df2.Model == 'sel600'), '600sel', df2['Model'])
    df2['Model'] = np.where((df2.Make == 'Mercedes-Benz') & (df2.Model == 'sl300'), '300sl', df2['Model'])


    # Output vehicle make/models to csv
    make_models_vmmr = df2[['Make', 'Model']].drop_duplicates(subset=["Make", "Model"]).sort_values(by=['Make', 'Model']).reset_index(drop=True)
    merged = make_models_stanford.merge(make_models_vmmr, indicator=True, how='outer').sort_values(by=['Make', 'Model']).reset_index(drop=True)
    merged._merge = np.where(merged._merge=='right_only', 'vmmr', merged._merge)
    merged._merge = np.where(merged._merge == 'left_only', 'stanford', merged._merge)
    merged.to_csv('./data/stanford_vmmr_unique_make_models.csv', index=False)

    # Move files to new directories
    lst = []
    for i in range(len(df2)):
        output = os.path.join(output_path, df2['Make'].iloc[i], df2['Model'].iloc[i], str(df2['Year'].iloc[i]))
        os.makedirs(output, exist_ok=True)  # will just add if already exits
        copy(df2['Orig Path'].iloc[i], output)  # copy file to new dest with orig name
        old = os.path.join(output, df2['Orig Path'].iloc[i].split('/')[-1])  # old name
        new = os.path.join(output, 'vmmr' + '_' + str(i) + '.' + df2['Orig Path'].iloc[i].split('/')[-1].split('.')[-1])
        os.rename(old, new)
        lst.append(new)

    df2 = pd.concat([df2, pd.DataFrame(lst, columns=['New Path'])], axis=1)

    df2 = df2[['Make', 'Model', 'Year', 'Orig Path', 'New Path']]

    df2.to_csv(os.path.join(stanford_data, '../', 'vmmr_car_directory.csv'), index=False)