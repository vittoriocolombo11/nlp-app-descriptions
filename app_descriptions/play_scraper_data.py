# This is the code I used to get all the app descriptions. Now play_scraper has changed: running it throws an error

import play_scraper
import time
from time import sleep

collections=['NEW_FREE','TOP_FREE','TOP_PAID','TOP_GROSSING','TRENDING']

CATEGORIES = [
    "ART_AND_DESIGN",
    "BOOKS_AND_REFERENCE",
    "BUSINESS",
    "COMICS",
    "EDUCATION",
    "ENTERTAINMENT",
    "FINANCE",
    "FOOD_AND_DRINK",
    "GAME",
    "HEALTH_AND_FITNESS",
    "HOUSE_AND_HOME",
    "MAPS_AND_NAVIGATION",
    "MEDICAL",
    "MUSIC_AND_AUDIO",
    "NEWS_AND_MAGAZINES",
    "PERSONALIZATION",
    "PHOTOGRAPHY",
    "PRODUCTIVITY",
    "SHOPPING",
    "SPORTS",
    "WEATHER"]

apps=[]
for cat in CATEGORIES:
    for topic in collections:
        for p in range(0,4):
            g = play_scraper.collection(collection=topic,
                               category = cat,
                               results = 100,
                               page = p, 
                               detailed = True, 
                               hl = 'en')
            apps.extend(g) 
            print(p)
            time.sleep(30)

app_price=[]
app_score=[]
app_name=[]
app_description=[]
app_category = []
for i in range(len(apps)):
    app_name.append(apps[i]['title'])
    app_description.append(apps[i]['description'])
    app_score.append(apps[i]['score'])
    app_price.append(apps[i]['price'])
    if apps[i]['category'] == []:
        app_category.append("NaN")
    else:
        app_category.append(apps[i]['category'][0])
        
df=pd.DataFrame({'app':app_name, 'description':app_description, 'category':app_category, 
                 'price':app_price, 'rating': app_score}).drop_duplicates()


df.to_csv("corpus1_raw.csv", sep = "\t")