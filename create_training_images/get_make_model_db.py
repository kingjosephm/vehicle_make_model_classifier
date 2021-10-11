import json
import urllib
import requests
import pandas as pd

"""
 See: https://www.back4app.com/database/back4app/car-make-model-dataset
 
 From there you can create a profile and fetch data from whichever database. Below draws from `Car_Model_List`, 
 querying all vehicle makes and models from 2000 onwards.
"""

if __name__ == '__main__':


    where = urllib.parse.quote_plus("""
    {
        "Make": {
            "$exists": true
        },
        "Model": {
            "$exists": true
        },
        "Category": {
            "$exists": true
        },
        "Year": {
            "$gte": 2000
        }
    }
    """)
    url = 'https://parseapi.back4app.com/classes/Carmodels_Car_Model_List?limit=1000000&order=Make,Model,Year&where=%s' % where
    headers = {
        'X-Parse-Application-Id': 'VC5aFi4lbNR2WgeXnlouemf2C66M6MbU1EJ34y35', # This is your app's application id
        'X-Parse-REST-API-Key': 'wYMv3H1YB2XE3tHB7cM5LZya7ezd86JDuVIDUPBV' # This is your app's REST API key
    }
    data = json.loads(requests.get(url, headers=headers, verify=False).content.decode('utf-8')) # Here you have the data that you need

    lst = []
    for x in range(len(data['results'])):
        lst.append([data['results'][x]['Make'], data['results'][x]['Model'], data['results'][x]['Year'], data['results'][x]['Category']])

    df = pd.DataFrame(lst, columns=['Make', 'Model', 'Year', 'Category'])

    df = df.sort_values(by=['Make', 'Model', 'Year']).reset_index(drop=True)

    # Drop Freightliner truck
    df = df.loc[df.Make != 'Freightliner']

    # Consolidate SRT back into Dodge (even though was own brand briefly)
    df['Make'] = df['Make'].str.replace("SRT", 'Dodge')

    df['Make'] = df.Make.str.replace("Ram", "RAM").str.replace("FIAT", 'Fiat').str.replace("MAZDA", 'Mazda')

    df.to_csv('./data/make_model_database.csv', index=False)