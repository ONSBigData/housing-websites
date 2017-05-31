__author__ = 'Karen Gask'

# Collect information from the Zoopla API (property listings only)

from requests import get
import json
from math import floor
import numpy as np
import pandas as pd

# Add .js to url gets data in JSON format
# Add page_size=100 to increase number of properties obtained. 100 is maximum
# Add listing_status="sale" for properties for sale. Alternative is "rent"

def get_zoopla():

    # Run first API call for first web page only
    # This gets number of web pages required for area and checks the API is working
    api = 'http://api.zoopla.co.uk/api/v1/property_listings.js?'
    area = 'area=' + area_name
    api_key= '&api_key=add_your_api_key_here'
    page_size = '&page_size=100'
    include_sold = '&include_sold=1'
    include_rented = '&include_rented=1'
    page_number = '&page_number='

    api_input = api + area + api_key + page_size + include_sold + include_rented
    response = get(api_input)

    # Check API has worked
    if response.status_code == 200:
        print("API call was successful")
    elif response.status_code == 400:
        print("Bad request for " + area_name + ". Check that this area is searchable on Zoopla website")
    elif response.status_code == 403:
        print("API call was forbidden. Reached maximum calls per hour")
    else:
        print("You've messed up your coding! HTTP status code is ", response.status_code)

    content_as_string = response.content.decode()
    # Decode response body from JSON with json.loads()
    content = json.loads(content_as_string)

    # Number of web pages required for this area
    web_pages = int(floor((content['result_count'] + 100) / 100))

    # Print number of properties in this area
    print("Number of properties in " + area_name, content['result_count'])

    # Run second API call for all pages for this area
    homes = pd.DataFrame()
    for i in range(1, web_pages+1):
        api_input = api + area + api_key + page_size + include_sold + include_rented + page_number + str(i)
        response = get(api_input)
        content_as_string = response.content.decode()
        content = json.loads(content_as_string)
        # It's a dictionary - want just the key=listing in the dictionary
        listing = content['listing']
        # This is a list of dictionaries
        # Convert this to a pandas data frame
        listing = pd.DataFrame(listing)
        # Append to data frame if there is more than one web page
        if i==1:
            homes = listing
        else:
            homes = homes.append(listing)

    # Keep selected columns and convert price, num_bathroom and num_bedrooms to floats for further analysis
    if 'new_home' in homes:
        homes = homes[['county','description','displayable_address','property_type','street_name','price','listing_id',
                       'new_home','num_bathrooms','num_bedrooms','latitude','longitude','listing_status','agent_name']]
    else:
        homes = homes[['county','description','displayable_address','property_type','street_name','price','listing_id',
                       'num_bathrooms','num_bedrooms','latitude','longitude','listing_status','agent_name']]
    homes[['price','num_bathrooms','num_bedrooms']] = homes[['price','num_bathrooms','num_bedrooms']].astype(float)

    return homes


# Run function for all postcodes around Exeter

homes = pd.DataFrame()
postcode_districts = ['EX1','EX2','EX3','EX4','EX5','EX6','EX7','EX8','EX9','EX10','EX11','EX12',
                      'EX13','EX14','EX15','EX16']
for i in postcode_districts:
    area_name = i
    temp = get_zoopla()
    if 'new_home' not in temp:
        temp['new_home'] = np.nan
    if i == 0:
        homes = temp
    else:
        homes = homes.append(temp)

print(len(homes))
print(homes.head())

exeter = homes

exeter.to_csv("/nas/data/Smart Meter data/Karen/exeter.csv", encoding='utf-8')


