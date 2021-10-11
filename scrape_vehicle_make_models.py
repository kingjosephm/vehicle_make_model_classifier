import os
import time
import io
import hashlib
import signal
import requests
from PIL import Image
from selenium import webdriver
import pandas as pd
import caffeine
import json
import shutil

pd.set_option('display.max_columns', 100)


"""
    Credit:
    https://github.com/Ladvien/deep_arcane/blob/main/1_get_images/scrap.py
    
    Note: 
    Requires chromedriver installed in PATH variable

"""

class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def fetch_image_urls(query: str, max_links_to_fetch: int, wd: webdriver, sleep_between_interactions: float = 0.1):

    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    start = time.time()
    while image_count < max_links_to_fetch:

        if (time.time() - start) / 60 > 10:  # if still searching for >10 min, break
            continue

        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)

        print(
            f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}"
        )

        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls
            actual_images = wd.find_elements_by_css_selector("img.n3VNCb")
            for actual_image in actual_images:
                if actual_image.get_attribute(
                    "src"
                ) and "http" in actual_image.get_attribute("src"):
                    image_urls.add(actual_image.get_attribute("src"))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            time.sleep(8)

            not_what_you_want_button = ""
            try:
                not_what_you_want_button = wd.find_element_by_css_selector(".r0zKGf")
            except:
                pass

            # If there are no more images return.
            if not_what_you_want_button:
                print("No more images available.")
                return image_urls

            load_more_button = wd.find_element_by_css_selector(".mye4qd")
            if load_more_button and not not_what_you_want_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls

def search_and_download(wd, query, output_path, number_images=5):

    res = fetch_image_urls(
        query,
        number_images,
        wd=wd,
    )

    if res is not None:

        # Save image sources
        if os.path.exists(os.path.join(output_path, 'image_sources.json')):
            with open(os.path.join(output_path, 'image_sources.json'), 'rb') as j:
                image_sources = json.load(j)
        else:
            image_sources = {}

        for url in res:
            try:
                print("Getting image")
                with timeout(2):
                    image_content = requests.get(url, verify=False).content

            except Exception as e:
                print(f"ERROR - Could not download {url} - {e}")

            try:
                image_file = io.BytesIO(image_content)
                image = Image.open(image_file).convert("RGB")
                file_path = os.path.join(
                    output_path, hashlib.sha1(image_content).hexdigest()[:10] + ".jpg"
                )
                with open(file_path, "wb") as f:
                    image.save(f, "JPEG", quality=85)
                print(f"SUCCESS - saved {url} - as {file_path}")

                image_sources[file_path.split('/')[-1]] = url

            except Exception as e:
                print(f"ERROR - Could not save {url} - {e}")

            with open(os.path.join(output_path, 'image_sources.json'), 'w') as j:
                json.dump(image_sources, j)
        else:
            print(f"Failed to return links for term: {query}")

if __name__ == '__main__':

    df = pd.read_csv('./data/make_model_database_mod.csv')
    rootOutput = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier/scraped_images'

    number_images = 100

    # Remove vehicle make-model-year rows if dir already exists on disk (in case successfully ran previously)
    lst = []
    for subdir, dirs, files in os.walk(rootOutput):
        for file in [i for i in files if 'jpg' in i or 'png' in i]:
            lst.append('/'.join(os.path.join(subdir, file).split('/')[-4:]))  # does not count empty subdirectories
    foo = pd.DataFrame(lst, columns=["Path"])
    foo['Make'] = foo['Path'].apply(lambda x: x.split('/')[0])
    foo['Model'] = foo['Path'].apply(lambda x: x.split('/')[1])
    foo['Year'] = foo['Path'].apply(lambda x: x.split('/')[2]).astype(int)
    foo['dir'] = foo['Path'].apply(lambda x: '/'.join(x.split('/')[:-1]))

    # See if any incomplete, i.e. ~30 less than number of images desired
    foo['count'] = foo.groupby(['Make', 'Model', 'Year'])['Path'].transform('count')
    incomplete_dirs = foo.loc[foo['count'] < number_images-30]['dir'].drop_duplicates().tolist()
    if incomplete_dirs:
        for x in incomplete_dirs:
            shutil.rmtree(os.path.join(rootOutput, x))
            foo = foo.loc[foo['dir'] != x].reset_index(drop=True)

    foo = foo[['Make', 'Model', 'Year']].drop_duplicates().reset_index(drop=True)

    df = df.merge(foo, on=['Make', 'Model', 'Year'], how='outer', indicator=True)
    df = df.loc[df._merge == 'left_only'].reset_index(drop=True)
    del df['_merge']

    wd = webdriver.Chrome()
    wd.get("https://google.com")

    for i in range(len(df)):
        query = df.iloc[i, 0] + ' ' + df.iloc[i, 1] + ' ' + df.iloc[i, 3] + ' ' + str(df.iloc[i, 4])

        output_path = os.path.join(rootOutput, df.iloc[i, 0], df.iloc[i, 2], str(df.iloc[i, 4]))
        os.makedirs(output_path, exist_ok=True)

        search_and_download(wd, query, output_path, number_images=100)

    caffeine.off()