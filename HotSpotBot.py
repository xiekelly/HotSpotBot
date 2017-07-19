
# coding: utf-8

# # Hot.Spot.Bot.

# ## 1. Fetch data from JSON and store in database

# In[1]:

# Get the NYC Free Wifi data
import requests

url = 'https://data.cityofnewyork.us/api/views/varh-9tsp/rows.json'
wifi_data = requests.get(url).json()['data']
wifi_data


# In[2]:

import MySQLdb as mdb
import sys

# Set up the database in which we will store the WiFi data
def connectDB():
    con = mdb.connect(host = 'localhost', 
                      user = 'root', 
                      passwd = 'dwdstudent2015', 
                      charset = 'utf8', use_unicode=True);
    return con

# Create a database
def createwifiDB(con, db_name):
    create_db_query = "CREATE DATABASE IF NOT EXISTS {0} DEFAULT CHARACTER SET 'utf8'".format(db_name)
    cursor = con.cursor()
    cursor.execute(create_db_query)
    cursor.close()
    pass

con = connectDB()
db_name = 'WiFi'
createwifiDB(con, db_name)


# In[3]:

# Create table for Wifi
def createwifiTable(con, db_name, table_name):
    cursor = con.cursor()
    create_table_query = '''CREATE TABLE IF NOT EXISTS {0}.{1} 
                                    (objectid int,
                                    boro_name varchar(250), 
                                    hotspot_type varchar(250), 
                                    provider varchar(250),
                                    location varchar(250),
                                    lat float,
                                    lon float,
                                    location_t varchar(250),
                                    remarks varchar(250),
                                    city varchar(250),
                                    ssid varchar(250), 
                                    PRIMARY KEY(objectid)
                                    )'''.format(db_name, table_name)
    cursor.execute(create_table_query)
    cursor.close()

wifi_table = 'wifi_hotspots'
createwifiTable(con, db_name, wifi_table)


# In[4]:

# Store wifi data
def storewifiData(con, wifi_data):
    db_name = 'WiFi'
    table_name = 'wifi_hotspots'
    for hotspot in wifi_data:
        objectid = hotspot[10]
        boro_name = hotspot[11]
        hotspot_type = hotspot[9]
        provider = hotspot[12]
        location = hotspot[14]
        lat = hotspot[15]
        lon = hotspot[16]
        location_t = hotspot[19]
        remarks = hotspot[20]
        city = hotspot[21]
        ssid = hotspot[22]
        insertwifi(con, db_name, table_name, 
                      objectid, boro_name, hotspot_type, provider, location, lat, lon, location_t, remarks, city, ssid)
    con.commit()
    return

def insertwifi(con, db_name, table_name, 
                  objectid, boro_name, hotspot_type, provider, location, lat, lon, location_t, remarks, city, ssid):
    query_template = '''INSERT IGNORE INTO {0}.{1}(objectid, 
                                                    boro_name, 
                                                    hotspot_type, 
                                                    provider, 
                                                    location, 
                                                    lat, 
                                                    lon, 
                                                    location_t, 
                                                    remarks, 
                                                    city, 
                                                    ssid) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'''.format(db_name, table_name)
    cursor = con.cursor()
    query_parameters = (objectid, boro_name, hotspot_type, provider, location, lat, lon, location_t, remarks, city, ssid)
    cursor.execute(query_template, query_parameters)
    cursor.close()

storewifiData(con, wifi_data)


# In[5]:

# Install the geocoder API
# http://geocoder.readthedocs.io/api.html

get_ipython().system('sudo python3 -m pip install -U geocoder')
import geocoder


# In[6]:

# Get current lat/lon based on address

address = "60 Washington Square S, New York, NY 10012"
g = geocoder.google(address)
current_location = g.latlng


# In[7]:

current_location


# In[8]:

# Connect to MySQL WiFi database

get_ipython().magic('reload_ext sql')
get_ipython().magic('sql mysql://root:dwdstudent2015@localhost:3306/WiFi?charset=utf8')


# In[9]:

# Get hotspot locations--returns a list of pairs in parentheses(x, y)

hotspot_location = get_ipython().magic('sql SELECT lat, lon FROM wifi_hotspots')
hotspot_id = get_ipython().magic('sql SELECT objectid FROM wifi_hotspots')


# In[10]:

# Calculate distance between current location and all the hotspot locations

# equation from https://gist.github.com/rochacbruno/2883505

import math
def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 3959 # radius of world in miles

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1))         * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = float("{0:.4f}".format(radius * c))

    return d

hotspot_distances = []
for hotspot in hotspot_location:
    hotspot_distance = distance(current_location, hotspot)
    if hotspot_distance <= 1:
        hotspot_info = hotspot_distance, hotspot_location.index(hotspot) # basically the unique fid_1
        #hotspot_distances.append(hotspot_distance)
        #hotspot_distances.append(hotspot_location.index(hotspot))
        hotspot_distances.append(hotspot_info)

hotspot_distances


# In[11]:

# NO SLACK BOT: Return all wifi hotspot locations within x distance

hotspot_location_names = get_ipython().magic('sql SELECT location FROM wifi_hotspots')
#hotspot_location_ts = %sql SELECT location_t FROM wifi_hotspots
hotspot_ssids = get_ipython().magic('sql SELECT ssid FROM wifi_hotspots')


# In[12]:

print('The nearest wifi hotspot locations are:')
for hotspot in hotspot_location:
    hotspot_distance = distance(current_location, hotspot)
    if hotspot_distance <= 0.1:
        hotspot_location_name = hotspot_location_names[hotspot_location.index(hotspot)]
        hotspot_ssid = hotspot_ssids[hotspot_location.index(hotspot)]
        print(str(hotspot_location_name).strip("(),'"), 'at', hotspot_distance, 'miles away. It is called', str(hotspot_ssid).strip("(),'"))


# ## 2. Conduct analyses on data using Pandas

# In[13]:

get_ipython().magic('matplotlib inline')
import requests
import json
import pandas as pd
import numpy as np
import matplotlib as plt


# In[14]:

# Import wifi dataset to Pandas dataframe
wifi_locations = pd.read_csv("NYC_Free_Public_WiFi_03292017.csv", encoding="utf-8", dtype="unicode")

# Peek at dataframe
wifi_locations


# In[15]:

wifi_locations.keys() # Peek at dataframe columns


# In[16]:

# Convert data types to numeric for appropriate columns
wifi_locations['LAT'] = pd.to_numeric(wifi_locations['LAT'])
wifi_locations['LON'] = pd.to_numeric(wifi_locations['LON'])
wifi_locations['CounDist'] = pd.to_numeric(wifi_locations['CounDist'])


# In[17]:

# Plot adjustments
plt.rcParams['figure.figsize'] = (15, 15) # Make the graph bigger
plt.rcParams.update({'font.size': 22}) # Make font bigger


# In[18]:

# Place wifi kiosks on scatterplot based on latitude and longitude

# Make the size of each point proportional to the size of available kiosks
# in the vicinity.
free = (wifi_locations["TYPE"] == 'Free') # Free wifi spots
limited = (wifi_locations["TYPE"] == 'Limited Free') # Limited use wifi spots

firstplot = wifi_locations[free].plot(kind='scatter', 
                               x='LON', 
                               y='LAT', 
                               color='DarkBlue', 
                               alpha=0.5, 
                               label='Free', 
                               s=10*(wifi_locations['CounDist']),
                                title='NYC WiFi Hotspots')

secondplot = wifi_locations[limited].plot(kind='scatter', 
                                   x='LON', 
                                   y='LAT', 
                                   color='Red', 
                                   ax = firstplot, 
                                   s = 10*(wifi_locations['CounDist']), alpha = 0.5,
                                   label = 'Limited Free')


# In[19]:

# Separate data for each borough

man = wifi_locations[wifi_locations["BORO"] == 'MN'] # Manhattan

free_man = (man["TYPE"] == 'Free') # Free wifi spots
limited_man = (man["TYPE"] == 'Limited Free') # Limited use wifi spots

firstplot_man = man[free_man].plot(kind='scatter', 
                               x='LON', 
                               y='LAT', 
                               color='DarkBlue', 
                               alpha=0.5, 
                               label='Free', 
                               s=10*(man['CounDist']),
                                title='Manhattan WiFi Hotspots')

secondplot_man = man[limited_man].plot(kind='scatter', 
                                   x='LON', 
                                   y='LAT', 
                                   color='Red', 
                                   ax = firstplot_man, 
                                   s = 10*(man['CounDist']), alpha = 0.5,
                                   label = 'Limited Free')


# In[20]:

brk = wifi_locations[wifi_locations["BORO"] == 'BK'] # Brooklyn

free_brk = (brk["TYPE"] == 'Free') # Free wifi spots
limited_brk = (brk["TYPE"] == 'Limited Free') # Limited use wifi spots

firstplot_brk = brk[free_brk].plot(kind='scatter', 
                               x='LON', 
                               y='LAT', 
                               color='DarkBlue', 
                               alpha=0.5, 
                               label='Free', 
                               s=10*(brk['CounDist']),
                                title='Brooklyn WiFi Hotspots')

secondplot_brk = brk[limited_brk].plot(kind='scatter', 
                                   x='LON', 
                                   y='LAT', 
                                   color='Red', 
                                   ax = firstplot_brk, 
                                   s = 10*(brk['CounDist']), alpha = 0.5,
                                   label = 'Limited Free')


# In[21]:

brx = wifi_locations[wifi_locations["BORO"] == 'BX'] # Bronx

free_brx = (brx["TYPE"] == 'Free') # Free wifi spots
limited_brx = (brx["TYPE"] == 'Limited Free') # Limited use wifi spots

firstplot_brx = brx[free_brx].plot(kind='scatter', 
                               x='LON', 
                               y='LAT', 
                               color='DarkBlue', 
                               alpha=0.5, 
                               label='Free', 
                               s=10*(brx['CounDist']),
                                title='Bronx WiFi Hotspots')

secondplot_brx = brx[limited_brx].plot(kind='scatter', 
                                   x='LON', 
                                   y='LAT', 
                                   color='Red', 
                                   ax = firstplot_brx, 
                                   s = 10*(brx['CounDist']), alpha = 0.5,
                                   label = 'Limited Free')


# In[22]:

qns = wifi_locations[wifi_locations["BORO"] == 'QU'] # Queens

free_qns = (qns["TYPE"] == 'Free') # Free wifi spots
limited_qns = (qns["TYPE"] == 'Limited Free') # Limited use wifi spots

firstplot_qns = qns[free_qns].plot(kind='scatter', 
                               x='LON', 
                               y='LAT', 
                               color='DarkBlue', 
                               alpha=0.5, 
                               label='Free', 
                               s=10*(qns['CounDist']),
                                title='Queens WiFi Hotspots')

secondplot_qns = qns[limited_qns].plot(kind='scatter', 
                                   x='LON', 
                                   y='LAT', 
                                   color='Red', 
                                   ax = firstplot_qns, 
                                   s = 10*(qns['CounDist']), alpha = 0.5,
                                   label = 'Limited Free')


# In[23]:

stn = wifi_locations[wifi_locations["BORO"] == 'SI'] # Staten Island

free_stn = (stn["TYPE"] == 'Free') # Free wifi spots
limited_stn = (wifi_locations["TYPE"] == 'Limited Free') # Limited use wifi spots

firstplot_stn = stn[free_stn].plot(kind='scatter', 
                               x='LON', 
                               y='LAT', 
                               color='DarkBlue', 
                               alpha=0.5, 
                               label='Free', 
                               s=10*(stn['CounDist']),
                                title='Staten Island WiFi Hotspots')

secondplot_stn = stn[limited_stn].plot(kind='scatter', 
                                   x='LON', 
                                   y='LAT', 
                                   color='Red', 
                                   ax = firstplot_stn, 
                                   s = 10*(stn['CounDist']), alpha = 0.5,
                                   label = 'Limited Free')


# In[24]:

# Plot bar graph by type of free wifi available for each borough
colors = ['darkblue','yellowgreen']

df1 = wifi_locations.groupby(['BoroName', 'TYPE'])['TYPE'].count().unstack('TYPE')
ax1 = df1[['Free', 'Limited Free']].plot.barh(title = "WiFi HotSpot Types by Borough", stacked=True, colors=colors)
ax1

# Label bar plot
for p in ax1.patches:
    ax1.annotate(str(p.get_width()), (p.get_x() + p.get_width(), p.get_y()), xytext=(5, 10), textcoords='offset points')


# In[25]:

# Plot pie graphs by wifi hotspot location types in each borough

import matplotlib.pyplot as mp
plt.rcParams.update({'font.size': 16}) # Adjust font

locationtype = wifi_locations["LOCATION_T"].unique()
boros = ['Manhattan', 'Brooklyn', 'Bronx', 'Queens', 'Staten Island']
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','orange','darkblue']

df2 = wifi_locations.groupby(['BoroName', 'LOCATION_T'])['LOCATION_T'].count().unstack('BoroName')
i = 0
for boro in boros:
    mp.figure(i)
    df2[boro].plot.pie(title = "WiFi HotSpots Location Types across Boroughs", figsize=(8, 8),                        colors=colors, shadow=True, autopct='%1.1f%%')
    i = i+1
    
mp.show()
    


# In[26]:

# Plot bar graph by wifi providers in each borough

boros = ['Manhattan', 'Brooklyn', 'Bronx', 'Queens', 'Staten Island']
colors = ['darkblue', 'lightcoral', 'lightskyblue','orange', 'yellowgreen']

df3 = wifi_locations.groupby(['BoroName', 'PROVIDER'])['BoroName'].count().unstack('BoroName')
ax3 = df3[boros].plot.barh(title = "WiFi HotSpots by Provider", stacked=True, color=colors)
ax3

# Label bar plot
for p in ax3.patches:
    ax3.annotate(str(p.get_width()), (p.get_x() + p.get_width(), p.get_y()), xytext=(5, 10), textcoords='offset points')


# In[27]:

get_ipython().system('sudo python3 -m pip install -U geopandas')


# In[28]:

# Create a chloropleth map for NYC population density by neighborhood
# from U.S. 2010 Census data

get_ipython().magic('matplotlib inline')
import requests
import json
import pandas as pd
import geopandas as gpd
import ast


# In[29]:

# Import NTA shapefiles data from NYC.gov website
# https://www1.nyc.gov/site/planning/data-maps/open-data/dwn-nynta.page
get_ipython().system("curl 'http://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/nynta/FeatureServer/0/query?where=1=1&outFields=*&outSR=4326&f=geojson' -o 'nyc_neighborhood_tabulation_areas.json'")


# In[30]:

# Process JSON file
geojson=open('nyc_neighborhood_tabulation_areas.json', 'r').read()
hoods_geojson = json.loads(geojson)["features"]
df_hoods = gpd.GeoDataFrame.from_features(hoods_geojson)
df_hoods.set_index(['NTACode'],inplace=True)
df_hoods.sort_index(inplace=True) # Sort by NTACode

# Peek at dataframe
df_hoods.head() # 195 neighborhoods in total


# In[31]:

# Import population density dataset to Pandas dataframe
NTAinfo = pd.read_csv("New_York_City_Population_By_Neighborhood_Tabulation_Areas.csv", encoding="utf-8", dtype="unicode")
NTAinfo = NTAinfo[ NTAinfo.Year=='2010' ] # Keep only 2010 Census data
NTAinfo['Population'] = pd.to_numeric(NTAinfo['Population'])
NTAinfo.rename(columns={'NTA Code' : 'NTACode'}, inplace=True) # Rename columns

# Peek at dataframe
NTAinfo.head()


# In[32]:

# Get population for each neighborhood area
df_population = NTAinfo
df_population.set_index(['NTACode'],inplace=True)
df_population.sort_index(inplace=True) # Sort by NTACode

# Peek at dataframe
df_population.head()


# In[33]:

# Generate map for NYC population density only
# by joining the two dataframes
df_hoods.join(df_population).plot(figsize=(15,7), 
                                            column='Population', 
                                            cmap='YlGnBu', 
                                            linewidth=0.2)


# In[34]:

# Get a list of the top populated neighborhoods in NYC in descending order
top_populated_hoods = NTAinfo
top_populated_hoods
top_populated_hoods.sort_values('Population', ascending=False, inplace=True)
top_populated_hoods.head(20)


# In[35]:

# Compare the list above to the top free-wifi-serviced neighborhoods
# in descending order
top_serviced_hoods = wifi_locations['NTAName'].value_counts()
top_serviced_hoods.head(20)


# ## 3. Setting up a Slack Bot

# In[36]:

import time
import re
import requests


# In[37]:

def message_matches(message_text):
    
    regex_expression = '.*@hotspot' 
    regex = re.compile(regex_expression)
    # Check if the message text matches the regex above
    match = regex.match(message_text)
    # returns true if the match is not None (ie the regex had a match)
    return match != None 


# In[38]:

import geocoder

def extract_location(message_text):
    regex_expression = 'I am at (.+), where are the closest WiFi hotspots?'
    regex= re.compile(regex_expression)
    matches = regex.finditer(message_text)
    for match in matches:
        location_entered = match.group(1)
    g = geocoder.google(location_entered)
    current_location = g.latlng
    return current_location


# In[39]:

import math
def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 3959 # radius of world in miles

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1))         * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = float("{0:.4f}".format(radius * c))
    return d


# In[40]:

get_ipython().system('sudo -H pip3 install -U sqlalchemy')
import pandas

from sqlalchemy import create_engine
engine = create_engine('mysql://root:dwdstudent2015@localhost:3306/WiFi?charset=utf8')
df = pandas.read_sql("SELECT objectid, location, lat, lon FROM wifi_hotspots", engine)


# In[41]:

def create_message(closest_locations, user_location):
    user_location = re.compile('(.+) New York, (.+)')
    match_location = user_location.match(message_text)
    if match_location != None:
    #if user_location != None:
        message = "The closest WiFi hotspots near you are:\n"
        df1 = df.sort_values('distance')
        closest_locations = df1.head(3)
        list_locations = closest_locations.values.tolist()

        i = 0
        for hotspot in list_locations: #for row in dataframe
            closest_hotspot_name = list_locations[i][1]
            closest_hotspot_distance = list_locations[i][4]
            i+=1
            message += "{a} is {b} miles away.\n".format(a=closest_hotspot_name, b=closest_hotspot_distance)
    else:
        message = "Please ask in the format of: @hotspot bot I am at 'street address', New York, NY, where are the closest WiFi hotspots?"
        
    return message


# In[42]:

import json

secrets_file = 'slack_secret.json'
f = open(secrets_file, 'r') 
content = f.read()
f.close()

auth_info = json.loads(content)
auth_token = auth_info["access_token"]
bot_user_id = auth_info["user_id"]

from slackclient import SlackClient
sc = SlackClient(auth_token)


# In[43]:

# Connect to the Real Time Messaging API of Slack and process the events

if sc.rtm_connect():
    # We are going to be polling the Slack API for recent events continuously
    while True:
        # We are going to wait 1 second between monitoring attempts
        time.sleep(1)
        # If there are any new events, we will get a response. If there are no events, the response will be empty
        response = sc.rtm_read()
        for item in response:
            # Check that the event is a message. If not, ignore and proceed to the next event.
            if item.get("type") != 'message':
                continue
                
            # Check that the message comes from a user. If not, ignore and proceed to the next event.
            if item.get("user") == None:
                continue
            
            # Check that the message is asking the bot to do something. If not, ignore and proceed to the next event.
            user_id = auth_info["user_id"]
            print("User ID:", user_id)
            message_text = item.get('text')
            print("Message text:", message_text)
            if message_matches(message_text):
                print("Returns True")
            else:
                print("Returns False")
                continue
                
            # Get the username of the user who asked the question
            response = sc.api_call("users.info", user=item["user"])
            username = response['user'].get('name')
            
            print("User who is asking:", username)
            
            # Extract the user's location from the user's message
            user_location = extract_location(message_text)
            print("User location:", user_location)
            
            if user_location == None:
                message ="Please ask in the format of: @hotspot bot I am at 'street address', New York, NY 'zip code', where are the closest WiFi hotspots?"
                sc.api_call("chat.postMessage", channel="#assignment2_bots", text=message)
                continue 
                
                
            df = pandas.read_sql("SELECT objectid, location, lat, lon FROM wifi_hotspots", engine)
            distances = [distance(user_location, (row[2], row[3]) ) for index, row in df.iterrows()]
            df['distance'] = distances
            hotspot_location = df['location']
            df.sort_values('distance')
            df = df.sort_values('distance')
            closest_locations = df.head(3)
            
            # Prepare the message that we will send back to the user
            message = create_message(closest_locations, user_location)

        
            # Post a response to the #bots channel
            sc.api_call("chat.postMessage", channel="#assignment2_bots", text=message)

