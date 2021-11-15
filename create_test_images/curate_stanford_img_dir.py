import pandas as pd
import os
import numpy as np

pd.options.display.max_colwidth = None
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 550)

if __name__ == '__main__':

    data_dir_path = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier/data_directories/MakeModelDirectory_Bboxes.csv'
    dir_df = pd.read_csv(data_dir_path, usecols=['Make', 'Model'])

    root = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier/stanford_car_data'

    # Create pd.DataFrame of vehicle make models based on directory structure
    lst = []
    for subdir, dirs, files in os.walk(root):
        for file in [i for i in files if 'jpg' in i or 'png' in i]:
            lst.append('/'.join(os.path.join(subdir, file).split('/')[-4:]))

    df = pd.DataFrame(lst, columns=['Source Path'])
    df['Source Path'] = df['Source Path'].apply(lambda x: '/'.join(x.split('/')[1:]))

    df['Make-Model'] = df['Source Path'].apply(lambda x: x.split('/')[1])
    df['Year'] = df['Make-Model'].str[-4:].astype(int)
    df = df.loc[df['Year'] >= 2000]
    df['Make-Model'] = df['Make-Model'].str[:-4]  # Remove year
    df['Make'] = df['Make-Model'].apply(lambda x: x.split(' ')[0])

    # Standardizing vehicle make
    df['Make'] = np.where(df['Make'] == 'Ram', 'RAM', df['Make'])
    df['Make'] = np.where(df['Make'] == 'FIAT', 'Fiat', df['Make'])
    df['Make'] = np.where(df['Make'] == 'Land', 'Land Rover', df['Make'])
    df['Make'] = np.where(df['Make'] == 'Infiniti', 'INFINITI', df['Make'])

    df = df.loc[df['Make'].isin(dir_df['Make'].unique().tolist())].reset_index(drop=True)
    df['Model'] = df['Make-Model'].apply(lambda x: ' '.join(x.split(' ')[1:])).str.strip()
    del df['Make-Model']

    # Restrict scraped image directory dataframe to overlapping vehicle makes
    dir_df = dir_df.loc[dir_df['Make'].isin(df['Make'].unique().tolist())].reset_index(drop=True)
    dir_df['Make-Model'] = dir_df['Make'] + ' ' + dir_df['Model']


    # Clean vehicle model
    df['Model'] = np.where(df['Model'] == 'Rover Range Rover SUV', 'Range Rover', df['Model'])
    fillers = ['Hatchback', 'Coupe', 'Sedan', 'SUV', 'Wagon', 'SuperCab', 'Crew Cab', 'Extended Cab', 'Regular Cab',
               'Quad Cab', 'SUPER Duty Crew Cab', 'Van', 'SRT-8', 'Minivan', 'Hybrid', 'Wagon Van', 'Convertible',
               'Club Cab', 'SRT8', 'SUT']
    for x in fillers:
        df['Model'] = df['Model'].str.replace(x, '')
        df['Model'] = df['Model'].str.strip()

    df['Model'] = np.where(df['Model'] == 'Ram Pickup 3500', 'Ram', df['Model'])
    df.loc[(df.Make == 'Chevrolet') & (df.Model.str.contains("Silverado")), 'Model'] = 'Silverado'
    df['Model'] = np.where(df['Model'] == 'Escalade EXT', 'Escalade', df['Model'])
    df['Model'] = np.where(df['Model'] == 'F-450 Super Duty', 'F-Series', df['Model'])
    df['Model'] = np.where(df['Model'] == 'F-150', 'F-Series', df['Model'])
    df.loc[(df.Make == 'Audi') & (df['Model'].str.contains("TT")), 'Model'] = 'TT'
    df['Model'] = np.where(df['Model'] == 'Corvette Ron Fellows Edition Z06', 'Corvette', df['Model'])
    df['Model'] = np.where(df['Model'] == 'Corvette ZR1', 'Corvette', df['Model'])
    df['Model'] = np.where(df['Model'] == 'Elantra Touring', 'Elantra', df['Model'])

    df.loc[(df.Make == 'Acura') & (df.Model.str.contains("TL")), 'Model'] = 'TL'
    df.loc[(df.Make == 'Buick') & (df.Model.str.contains("Regal")), 'Model'] = 'Regal'
    df.loc[(df.Make == 'Chevrolet') & (df.Model.str.contains("Express")), 'Model'] = 'Express'
    df.loc[(df.Make == 'Chevrolet') & (df.Model.str.contains("TrailBlazer")), 'Model'] = 'Trailblazer'
    df.loc[(df.Make == 'Dodge') & (df.Model.str.contains("Sprinter")), 'Model'] = 'Sprinter'
    df.loc[(df.Make == 'Fiat') & (df.Model.str.contains("500")), 'Model'] = '500'
    df.loc[(df.Make == 'Ford') & (df.Model.str.contains("Expedition")), 'Model'] = 'Expedition'
    df.loc[(df.Make == 'INFINITI') & (df.Model.str.contains("G")), 'Model'] = 'G'
    df.loc[(df.Make == 'Land Rover') & (df.Model.str.contains("LR2")), 'Model'] = 'LR2'
    df.loc[(df.Make == 'MINI') & (df.Model.str.contains("Cooper")), 'Model'] = 'Cooper'
    df.loc[(df.Make == 'RAM') & (df.Model.str.contains("C-V")), 'Model'] = 'C/V'
    df.loc[(df.Make == 'Jeep') & (df.Model.str.contains("Cherokee")), 'Model'] = 'Cherokee'
    df.loc[(df.Make == 'Jaguar') & (df.Model == "XK XKR"), 'Model'] = 'XK'
    df['Model'] = np.where(df['Model'] == 'Integra Type R', 'Integra', df['Model'])
    df['Model'] = np.where(df['Model'] == 'Town and Country', 'Town & Country', df['Model'])
    df['Model'] = np.where(df['Model'] == 'Cobalt SS', 'Cobalt', df['Model'])
    df['Model'] = np.where(df['Model'] == 'Leaf', 'LEAF', df['Model'])
    df['Model'] = np.where(df['Model'] == 'Juke', 'JUKE', df['Model'])
    df['Model'] = np.where(df['Model'] == 'NV Passenger', 'NV-Series', df['Model'])
    df.loc[(df.Make == 'Cadillac') & (df.Model == "CTS-V"), 'Model'] = 'CTS'
    df.loc[(df.Make == 'Mercedes-Benz') & (df.Model == "SL-Class"), 'Model'] = 'SL'
    df['Model'] = np.where(df['Model'] == 'HHR SS', 'HHR', df['Model'])

    df['Model'] = df['Model'].str.strip()

    # Restrict to overlapping Make-Models
    keepers = dir_df['Make-Model'].unique().tolist()

    df['Make-Model'] = df['Make'] + ' ' + df['Model']
    df = df.loc[df['Make-Model'].isin(keepers)].reset_index(drop=True)

    df = df[['Make', 'Model', 'Make-Model', 'Year', 'Source Path']]
    df = df.sample(frac=1, random_state=123)
    df.to_csv(os.path.join(root, 'stanford_img_dir.csv'), index=False)