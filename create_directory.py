import pandas as pd
import os
import shutil
pd.set_option('display.max_columns', 500)

def fix_dir_structure(rootDir):

    # Renames subdirs, removing vehicle make from subdir name
    # This was done after manually moving all make/model subdirs into parent make dir
    dirs = sorted([i for i in os.listdir(rootDir) if ".DS_Store" not in i])
    for i in dirs:
        parent = os.path.join(path, i).split('/')[-1] # car brand
        for j in sorted([i for i in os.listdir(os.path.join(rootDir, i)) if ".DS_Store" not in i]):
            renamed = ' '.join(j.split('_')).replace(parent, '').strip()
            os.rename(os.path.join(rootDir, i, j), os.path.join(rootDir, i, renamed))
    del i, parent, j


    # Remove years from model subdir title
    dirs = sorted([i for i in os.listdir(rootDir) if ".DS_Store" not in i])
    for i in dirs:
        for j in sorted([i for i in os.listdir(os.path.join(rootDir, i)) if ".DS_Store" not in i]):
            current = os.path.join(rootDir, i, j)
            subdir = os.path.join(rootDir, i, j, j[-9:])
            os.makedirs(subdir, exist_ok=True) # create new empty year subdir
            for file in [i for i in os.listdir(current) if '.jpg' in i or '.png' in i]:
                shutil.move(os.path.join(current, file), subdir)
            os.rename(current, os.path.join(rootDir, i, j[:-10]))
    del i, j


if __name__ == '__main__':

    path = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier/data'

    #fix_dir_structure(path)

    # Create pd.DataFrame of vehicle make models based on directory structure
    lst = []
    for subdir, dirs, files in os.walk(path):
        for file in [i for i in files if 'jpg' in i or 'png' in i]:
            lst.append('/'.join(os.path.join(subdir, file).split('/')[-4:]))

    df = pd.DataFrame(lst, columns=['Path'])

    # Add columns
    df['Make'] = df['Path'].apply(lambda x: ''.join(x.split('/')[0]))
    df['Model'] = df['Path'].apply(lambda x: ''.join(x.split('/')[1]))
    df['Years'] = df['Path'].apply(lambda x: ''.join(x.split('/')[2]))

    # Rearrange
    df = df[['Make', 'Model', 'Years', 'Path']]
    df = df.sort_values(by=['Make', 'Model', 'Years']).reset_index(drop=True)

    df.to_csv(os.path.join(path, '../' 'Vehicle Make Model Directory.csv'), index=False)

