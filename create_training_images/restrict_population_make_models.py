import pandas as pd
import json
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)

if __name__ == '__main__':

    df = pd.read_csv('make_model_database.csv')

    with open('utils/make_modeL_database_clean.json') as f:
        fixes = json.load(f)

    # Restrict to brands of interest
    keep_models = ['Acura', 'Alfa Romeo', 'Aston Martin', 'Audi', 'BMW', 'Bentley',
       'Buick', 'Cadillac', 'Chevrolet', 'Chrysler', 'Daewoo', 'Dodge',
       'Fiat', 'Ferrari', 'Ford', 'GMC', 'Genesis', 'HUMMER',
       'Honda', 'Hyundai', 'INFINITI', 'Isuzu', 'Jaguar', 'Jeep', 'Kia',
       'Lamborghini', 'Land Rover', 'Lexus', 'Lincoln', 'Lotus', 'Mazda',
       'MINI', 'Maserati', 'Maybach', 'McLaren', 'Mercedes-Benz',
       'Mercury', 'Mitsubishi', 'Nissan', 'Oldsmobile',
       'Plymouth', 'Pontiac', 'Porsche', 'RAM',
       'Rolls-Royce', 'Saab', 'Saturn', 'Scion', 'Subaru', 'Suzuki',
       'Tesla', 'Toyota', 'Volkswagen', 'Volvo', 'smart']  # Excludes Fisker, Polestar, Panoz, Rivian

    df = df.loc[df.Make.isin(keep_models)].reset_index(drop=True)

    # Fix errors in category
    df['Category'] = df['Category'].str.replace('Van/Minivan', 'Van')
    df.loc[df.Category.str.contains('SUV'), 'Category'] = 'SUV'

    # Expand number of make-model-year combinations if e.g. sedan and hatchback variants
    drop_indices = []
    for i in range(len(df)):
        if len(df.iloc[i, 3].split(',')) > 1:
            drop_indices.append(i)
            for x in df.iloc[i, 3].split(','):
                temp = pd.DataFrame({"Make": df.iloc[i, 0], "Model": df.iloc[i, 1], "Year": df.iloc[i, 2], "Category": x.strip()}, index=[100000 + i])
                df = pd.concat([df, temp], axis=0)

    df.drop(index=drop_indices, inplace=True)  # drop original index containing >1 categories
    df = df.sort_values(by=['Make', 'Model', 'Year', 'Category']).reset_index(drop=True)

    df['Detailed Model'] = df['Model'].copy()

    for key, val in fixes.items():
        df['Model'] = df['Model'].str.replace(rf'\b{key}\b', val, regex=True)
        df['Model'] = df['Model'].apply(lambda x: ' '.join(x.split())) # remove whitespace, if above created any


    # Acura
    df.loc[(df.Make == 'Acura') & (df['Model'].str.contains("MDX")), 'Model'] = 'MDX'
    df.loc[(df.Make == 'Acura') & (df['Model'].str.contains("RLX")), 'Model'] = 'RLX'

    # Alfa Romeo
    df.loc[(df.Make == 'Alfa Romeo') & (df['Model'].str.contains("4C")), 'Model'] = '4C'

    # Aston Martin
    df.loc[(df.Make == 'Aston Martin') & (df['Model'].str.contains("DB9")), 'Model'] = 'DB9'
    df.loc[(df.Make == 'Aston Martin') & (df['Model'].str.contains("Rapide")), 'Model'] = 'Rapide'
    df.loc[(df.Make == 'Aston Martin') & (df['Model'].str.contains("Vanquish")), 'Model'] = 'Vanquish'

    # Audi
    df.loc[(df.Make == 'Audi') & (df['Model'].str.contains("A3")), 'Model'] = 'A3'
    df.loc[(df.Make == 'Audi') & (df['Model'].str.contains("A4")), 'Model'] = 'A4'
    df.loc[(df.Make == 'Audi') & (df['Model'].str.contains("A5")), 'Model'] = 'A5'
    df.loc[(df.Make == 'Audi') & (df['Model'].str.contains("A6")), 'Model'] = 'A6'
    df.loc[(df.Make == 'Audi') & (df['Model'].str.contains("S4")), 'Model'] = 'S4'

    # BMW
    df.loc[(df.Make == 'BMW') & (df['Model'].str.contains("X4")), 'Model'] = 'X4'
    df.loc[(df.Make == 'BMW') & (df['Model'].str.contains("X5")), 'Model'] = 'X5'
    df.loc[(df.Make == 'BMW') & (df['Model'].str.contains("X6")), 'Model'] = 'X6'
    df.loc[(df.Make == 'BMW') & (df['Model'].str.contains("Z4")), 'Model'] = 'Z4'

    # Bentley
    df.loc[(df.Make == 'Bentley') & (df['Model'].str.contains("Azure")), 'Model'] = 'Azure'

    # Buick
    df.loc[(df.Make == 'Buick') & (df['Model'].str.contains("Encore")), 'Model'] = 'Encore'
    df.loc[(df.Make == 'Buick') & (df['Model'].str.contains("Regal")), 'Model'] = 'Regal'

    # Cadillac
    df.loc[(df.Make == 'Cadillac') & (df['Model'].str.contains("ATS")), 'Model'] = 'ATS'
    df.loc[(df.Make == 'Cadillac') & (df['Model'].str.contains("CT6")), 'Model'] = 'CT6'
    df.loc[(df.Make == 'Cadillac') & (df['Model'].str.contains("CTS")), 'Model'] = 'CTS'
    df.loc[(df.Make == 'Cadillac') & (df['Model'].str.contains("Escalade")), 'Model'] = 'Escalade'

    # Chevrolet
    df.loc[(df.Make == 'Chevrolet') & (df['Model'].str.contains("Silverado")), 'Model'] = 'Silverado'
    df.loc[(df.Make == 'Chevrolet') & (df['Model'].str.contains("Suburban")), 'Model'] = 'Suburban'
    df.loc[(df.Make == 'Chevrolet') & (df['Model'].str.contains("Tahoe")), 'Model'] = 'Tahoe'
    df.loc[(df.Make == 'Chevrolet') & (df['Model'].str.contains("Malibu")), 'Model'] = 'Malibu'
    df.loc[(df.Make == 'Chevrolet') & (df['Model'].str.contains("Colorado")), 'Model'] = 'Colorado'
    df.loc[(df.Make == 'Chevrolet') & (df['Model'].str.contains("Avalanche")), 'Model'] = 'Avalanche'
    df.loc[(df.Make == 'Chevrolet') & (df['Model'].str.contains("Uplander")), 'Model'] = 'Uplander'
    df.loc[(df.Make == 'Chevrolet') & (df['Model'].str.contains("Venture")), 'Model'] = 'Venture'
    df.loc[(df.Make == 'Chevrolet') & (df['Model'].str.contains("Astro")), 'Model'] = 'Astro'
    df.loc[(df.Make == 'Chevrolet') & (df['Model'].str.contains("2500")), 'Model'] = 'C/K'
    df.loc[(df.Make == 'Chevrolet') & (df['Model'].str.contains("3500")), 'Model'] = 'C/K'
    df.loc[(df.Make == 'Chevrolet') & (df['Model'].str.contains("Express")), 'Model'] = 'Express'

    # Chrysler
    df.loc[(df.Make == 'Chrysler') & (df['Model'].str.contains("Pacifica")), 'Model'] = 'Pacifica'
    df.loc[(df.Make == 'Chrysler') & (df['Model'].str.contains("Voyager")), 'Model'] = 'Voyager'

    # Dodge
    df.loc[(df.Make == 'Dodge') & (df['Model'].str.contains("Dakota")), 'Model'] = 'Dakota'
    df.loc[(df.Make == 'Dodge') & (df['Model'].str.contains("Ram Van")), 'Model'] = 'Ram Van'
    df.loc[(df.Make == 'Dodge') & (df['Model'].str.contains("Ram Wagon")), 'Model'] = 'Ram Van'
    df.loc[(df.Make == 'Dodge') & (df['Model'].str.contains("Sprinter")), 'Model'] = 'Sprinter'
    df.loc[(df.Make == 'Dodge') & (df['Model'].str.contains("Ram") & (df['Model'].str.contains("00"))), 'Model'] = 'Ram'
    df.loc[(df.Make == 'Dodge') & (df['Model'].str.contains("Caravan")), 'Model'] = 'Caravan'

    # Ferrari
    df.loc[(df.Make == 'Ferrari') & (df['Model'].str.contains("458")), 'Model'] = '458'

    # Fiat
    df.loc[(df.Make == 'Fiat') & (df['Model'].str.contains("500")), 'Model'] = '500'

    # Ford
    df.loc[(df.Make == 'Ford') & (df['Model'].str.contains("C-MAX")), 'Model'] = 'C-MAX'
    df.loc[(df.Make == 'Ford') & (df['Model'].str.contains("E150")), 'Model'] = 'E-Series'
    df.loc[(df.Make == 'Ford') & (df['Model'].str.contains("E250")), 'Model'] = 'E-Series'
    df.loc[(df.Make == 'Ford') & (df['Model'].str.contains("E350")), 'Model'] = 'E-Series'
    df.loc[(df.Make == 'Ford') & (df['Model'].str.contains("Econoline")), 'Model'] = 'E-series'
    df.loc[(df.Make == 'Ford') & (df['Model'].str.contains("Expedition")), 'Model'] = 'Expedition'
    df.loc[(df.Make == 'Ford') & (df['Model'].str.contains("Explorer")), 'Model'] = 'Explorer'
    df.loc[(df.Make == 'Ford') & (df['Model'].str.contains("F150")), 'Model'] = 'F-Series'
    df.loc[(df.Make == 'Ford') & (df['Model'].str.contains("F250")), 'Model'] = 'F-Series'
    df.loc[(df.Make == 'Ford') & (df['Model'].str.contains("F350")), 'Model'] = 'F-Series'
    df.loc[(df.Make == 'Ford') & (df['Model'].str.contains("F450")), 'Model'] = 'F-Series'
    df.loc[(df.Make == 'Ford') & (df['Model'].str.contains("Focus")), 'Model'] = 'Focus'
    df.loc[(df.Make == 'Ford') & (df['Model'].str.contains("Freestar")), 'Model'] = 'Freestar'
    df.loc[(df.Make == 'Ford') & (df['Model'].str.contains("Fusion")), 'Model'] = 'Fusion'
    df.loc[(df.Make == 'Ford') & (df['Model'].str.contains("Ranger")), 'Model'] = 'Ranger'
    df.loc[(df.Make == 'Ford') & (df['Model'].str.contains("Taurus")), 'Model'] = 'Taurus'
    df.loc[(df.Make == 'Ford') & (df['Model'].str.contains("Transit")), 'Model'] = 'Transit'
    df.loc[(df.Make == 'Ford') & (df['Model'].str.contains("Windstar")), 'Model'] = 'Windstar'

    # GMC
    df.loc[(df.Make == 'GMC') & (df['Model'].str.contains("Acadia")), 'Model'] = 'Acadia'
    df.loc[(df.Make == 'GMC') & (df['Model'].str.contains("Canyon")), 'Model'] = 'Canyon'
    df.loc[(df.Make == 'GMC') & (df['Model'].str.contains("Envoy")), 'Model'] = 'Envoy'
    df.loc[(df.Make == 'GMC') & (df['Model'].str.contains("Safari")), 'Model'] = 'Safari'
    df.loc[(df.Make == 'GMC') & (df['Model'].str.contains("Savana")), 'Model'] = 'Savana'
    df.loc[(df.Make == 'GMC') & (df['Model'].str.contains("Sierra")), 'Model'] = 'Sierra'
    df.loc[(df.Make == 'GMC') & (df['Model'].str.contains("Sonoma")), 'Model'] = 'Sonoma'
    df.loc[(df.Make == 'GMC') & (df['Model'].str.contains("Yukon")), 'Model'] = 'Yukon'

    # Honda
    df.loc[(df.Make == 'Honda') & (df['Model'].str.contains("Civic")), 'Model'] = 'Civic'
    df.loc[(df.Make == 'Honda') & (df['Model'].str.contains("Clarity")), 'Model'] = 'Clarity'

    # Hyundai
    df.loc[(df.Make == 'Hyundai') & (df['Model'].str.contains("Elantra")), 'Model'] = 'Elantra'
    df.loc[(df.Make == 'Hyundai') & (df['Model'].str.contains("Ioniq")), 'Model'] = 'Ioniq'
    df.loc[(df.Make == 'Hyundai') & (df['Model'].str.contains("Kona")), 'Model'] = 'Kona'
    df.loc[(df.Make == 'Hyundai') & (df['Model'].str.contains("Santa Fe")), 'Model'] = 'Santa Fe'
    df.loc[(df.Make == 'Hyundai') & (df['Model'].str.contains("Sonata")), 'Model'] = 'Sonata'
    df.loc[(df.Make == 'Hyundai') & (df['Model'].str.contains("Tucson")), 'Model'] = 'Tucson'
    df.loc[(df.Make == 'Hyundai') & (df['Model'].str.contains("Tucson")), 'Model'] = 'Tucson'
    df.loc[(df.Make == 'Hyundai') & (df['Model'].str.contains("XG350")), 'Model'] = 'XG'
    df.loc[(df.Make == 'Hyundai') & (df['Model'].str.contains("XG300")), 'Model'] = 'XG'
    df.loc[(df.Make == 'Hyundai') & (df['Model'].str.contains("Genesis")), 'Model'] = 'Genesis'

    # Isuzu
    df.loc[(df.Make == 'Isuzu') & (df['Model'].str.contains("Hombre")), 'Model'] = 'Hombre'
    df.loc[(df.Make == 'Isuzu') & (df['Model'].str.contains("Rodeo")), 'Model'] = 'Rodeo'
    df.loc[(df.Make == 'Isuzu') & (df['Model'].str.contains("i-")), 'Model'] = 'i-Series'

    # Jeep
    df.loc[(df.Make == 'Jeep') & (df['Model'].str.contains("Cherokee")), 'Model'] = 'Cherokee'
    df.loc[(df.Make == 'Jeep') & (df['Model'].str.contains("Wrangler")), 'Model'] = 'Wrangler'

    # Kia
    df.loc[(df.Make == 'Kia') & (df['Model'].str.contains("Forte")), 'Model'] = 'Forte'
    df.loc[(df.Make == 'Kia') & (df['Model'].str.contains("Niro")), 'Model'] = 'Niro'
    df.loc[(df.Make == 'Kia') & (df['Model'].str.contains("Optima")), 'Model'] = 'Optima'
    df.loc[(df.Make == 'Kia') & (df['Model'].str.contains("Soul")), 'Model'] = 'Soul'

    # Lamborghini
    df.loc[(df.Make == 'Lamborghini') & (df['Model'].str.contains("Murcielago")), 'Model'] = 'Murcielago'

    # Land Rover
    df.loc[(df.Make == 'Land Rover') & (df['Model'].str.contains("Discovery")), 'Model'] = 'Discovery'
    df.loc[(df.Make == 'Land Rover') & (df['Model'].str.contains("Range Rover")), 'Model'] = 'Range Rover'

    # Lexus
    df.loc[(df.Make == 'Lexus') & (df['Model'].str.contains("IS")), 'Model'] = 'IS'

    # Lincoln
    df.loc[(df.Make == 'Lincoln') & (df['Model'].str.contains("Navigator")), 'Model'] = 'Navigator'

    # Lotus
    df.loc[(df.Make == 'Lotus') & (df['Model'].str.contains("Evora")), 'Model'] = 'Evora'
    df.loc[(df.Make == 'Lotus') & (df['Model'].str.contains("Exige")), 'Model'] = 'Exige'

    # MINI
    df.loc[(df.Make == 'MINI') & (df['Model'].str.contains("Hardtop")), 'Model'] = 'Hardtop'

    # Mazda
    df.loc[(df.Make == 'Mazda') & (df['Model'].str.contains("MX-5 Miata")), 'Model'] = 'MX-5 Miata'
    df.loc[(df.Make == 'Mazda') & (df['Model'].str.contains("B-Series")), 'Model'] = 'B-Series'

    # Mercedes
    df.loc[(df.Make == 'Mercedes-Benz') & (df['Model'].str.contains("CLA-Class")), 'Model'] = 'CLA'  # standardizing onto more recent convention
    df.loc[(df.Make == 'Mercedes-Benz') & (df['Model'].str.contains("CLS-Class")), 'Model'] = 'CLS'
    df.loc[(df.Make == 'Mercedes-Benz') & (df['Model'].str.contains("GLA-Class")), 'Model'] = 'GLA'
    df.loc[(df.Make == 'Mercedes-Benz') & (df['Model'].str.contains("GLC Coupe")), 'Model'] = 'GLC'
    df.loc[(df.Make == 'Mercedes-Benz') & (df['Model'].str.contains("GLE-Coupe")), 'Model'] = 'GLE'
    df.loc[(df.Make == 'Mercedes-Benz') & (df['Model'].str.contains("GLE Coupe")), 'Model'] = 'GLE'
    df.loc[(df.Make == 'Mercedes-Benz') & (df['Model'].str.contains("SL-Class")), 'Model'] = 'SL'
    df.loc[(df.Make == 'Mercedes-Benz') & (df['Model'].str.contains("SLK-Class")), 'Model'] = 'SLK'
    df.loc[(df.Make == 'Mercedes-Benz') & (df['Model'].str.contains("Sprinter")), 'Model'] = 'Sprinter'
    df.loc[(df.Make == 'Mercedes-Benz') & (df['Model'].str.contains("Metris")), 'Model'] = 'Metris'

    # Mitsubishi
    df.loc[(df.Make == 'Mitsubishi') & (df['Model'].str.contains("Lancer")), 'Model'] = 'Lancer'
    df.loc[(df.Make == 'Mitsubishi') & (df['Model'].str.contains("Mirage")), 'Model'] = 'Mirage'
    df.loc[(df.Make == 'Mitsubishi') & (df['Model'].str.contains("Montero")), 'Model'] = 'Montero'
    df.loc[(df.Make == 'Mitsubishi') & (df['Model'].str.contains("Outlander")), 'Model'] = 'Outlander'
    df.loc[(df.Make == 'Mitsubishi') & (df['Model'].str.contains("Raider")), 'Model'] = 'Raider'

    # Nissan
    df.loc[(df.Make == 'Nissan') & (df['Model'].str.contains("Frontier")), 'Model'] = 'Frontier'
    df.loc[(df.Make == 'Nissan') & (df['Model'].str.contains("NV")), 'Model'] = 'NV-Series'
    df.loc[(df.Make == 'Nissan') & (df['Model'].str.contains("Pathfinder")), 'Model'] = 'Pathfinder'
    df.loc[(df.Make == 'Nissan') & (df['Model'].str.contains("Rogue")), 'Model'] = 'Rogue'
    df.loc[(df.Make == 'Nissan') & (df['Model'].str.contains("Titan", case=False)), 'Model'] = 'Titan'
    df.loc[(df.Make == 'Nissan') & (df['Model'].str.contains("Versa")), 'Model'] = 'Versa'

    # Plymouth
    df.loc[(df.Make == 'Plymouth') & (df['Model'].str.contains("Voyager")), 'Model'] = 'Voyager'

    # Pontiac
    df.loc[(df.Make == 'Pontiac') & (df['Model'].str.contains("2009.5")), 'Model'] = 'G6'
    df.loc[(df.Make == 'Pontiac') & (df['Model'].str.contains("Montana")), 'Model'] = 'Montana'

    # Porsche
    df.loc[(df.Make == 'Porsche') & (df['Model'].str.contains("Cayenne")), 'Model'] = 'Cayenne'

    # RAM
    df.loc[(df.Make == 'RAM') & (df['Model'].str.contains("1500")), 'Model'] = 'Pickup'
    df.loc[(df.Make == 'RAM') & (df['Model'].str.contains("2500")), 'Model'] = 'Pickup'
    df.loc[(df.Make == 'RAM') & (df['Model'].str.contains("3500")), 'Model'] = 'Pickup'
    df.loc[(df.Make == 'RAM') & (df['Model'].str.contains("C/V")), 'Model'] = 'C/V'
    df.loc[(df.Make == 'RAM') & (df['Model'].str.contains("Dakota")), 'Model'] = 'Dakota'
    df.loc[(df.Make == 'RAM') & (df['Model'].str.contains("ProMaster")), 'Model'] = 'ProMaster'

    # Suzuki
    df.loc[(df.Make == 'Suzuki') & (df['Model'].str.contains("Equator")), 'Model'] = 'Equator'
    df.loc[(df.Make == 'Suzuki') & (df['Model'].str.contains("XL-7")), 'Model'] = 'XL7'  # Standardize to later years

    # Toyota
    df.loc[(df.Make == 'Toyota') & (df['Model'].str.contains("Avalon")), 'Model'] = 'Avalon'
    df.loc[(df.Make == 'Toyota') & (df['Model'].str.contains("Camry")), 'Model'] = 'Camry'
    df.loc[(df.Make == 'Toyota') & (df['Model'].str.contains("Corolla")), 'Model'] = 'Corolla'
    df.loc[(df.Make == 'Toyota') & (df['Model'].str.contains("Highlander")), 'Model'] = 'Highlander'
    df.loc[(df.Make == 'Toyota') & (df['Model'].str.contains("Prius")), 'Model'] = 'Prius'
    df.loc[(df.Make == 'Toyota') & (df['Model'].str.contains("RAV4")), 'Model'] = 'RAV4'
    df.loc[(df.Make == 'Toyota') & (df['Model'].str.contains("Tacoma")), 'Model'] = 'Tacoma'
    df.loc[(df.Make == 'Toyota') & (df['Model'].str.contains("Tundra")), 'Model'] = 'Tundra'
    df.loc[(df.Make == 'Toyota') & (df['Model'].str.contains("Yaris")), 'Model'] = 'Yaris'

    # Volkswagen
    df.loc[(df.Make == 'Volkswagen') & (df['Model'].str.contains("Atlas")), 'Model'] = 'Atlas'
    df.loc[(df.Make == 'Volkswagen') & (df['Model'].str.contains("Golf")), 'Model'] = 'Golf'
    df.loc[(df.Make == 'Volkswagen') & (df['Model'].str.contains("Jetta")), 'Model'] = 'Jetta'
    df.loc[(df.Make == 'Volkswagen') & (df['Model'].str.contains("Beetle")), 'Model'] = 'Beetle'
    df.loc[(df.Make == 'Volkswagen') & (df['Model'].str.contains("Passat")), 'Model'] = 'Passat'
    df.loc[(df.Make == 'Volkswagen') & (df['Model'].str.contains("Tiguan")), 'Model'] = 'Tiguan'
    df.loc[(df.Make == 'Volkswagen') & (df['Model'].str.contains("Touareg")), 'Model'] = 'Touareg'

    # Volvo
    df.loc[(df.Make == 'Volvo') & (df['Model'].str.contains("S40")), 'Model'] = 'S40'

    # smart
    df.loc[(df.Make == 'smart') & (df['Model'].str.contains("fortwo")), 'Model'] = 'fortwo'


    df = df[['Make', 'Detailed Model', 'Model', 'Category', 'Year']]

    df = df.sort_values(by=['Make', 'Detailed Model', 'Year']).reset_index(drop=True)

    df.to_csv('./data/make_model_database_mod.csv', index=False)