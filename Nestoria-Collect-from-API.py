__author__ = 'Karen'

# Collect as much information from the Nestoria API (property listings only)

from requests import get
import json
import pandas as pd

def get_nestoria(type):

    # type must be 'buy' or 'rent'

    # Run first API call for first web page only
    # This gets number of web pages required for area and checks the API is working
    api = 'http://api.nestoria.co.uk/api?action=search_listings'
    place = '&place_name=' + area_name
    listing_type = '&listing_type=' + type
    json_uk = '&encoding=json&pretty=1&country=uk'
    page = '&page='

    api_input = api + place + listing_type + json_uk
    response = get(api_input)

    # Check API has worked
    if response.status_code == 200:
        print("API call was successful")
    elif response.status_code == 400:
        print("Bad request for " + area_name + ". Check that this area is searchable on Nestoria website")
    elif response.status_code == 403:
        print("API call was forbidden. Reached maximum calls")
    else:
        print("You've messed up your coding! HTTP status code is ", response.status_code)

    content_as_string = response.content.decode()
    # Decode response body from JSON with json.loads()
    content = json.loads(content_as_string)
    content_response = content['response']

    # Number of web pages required for this area
    web_pages = content_response['total_pages']

    # Print number of properties in this area
    print("Number of properties in " + area_name, content_response['total_results'])

    # Run second API call for all pages for this area
    homes = pd.DataFrame()
    for i in range(1, web_pages+1):
        api_input = api + place + listing_type + json_uk + page + str(i)
        response = get(api_input)
        content_as_string = response.content.decode()
        content = json.loads(content_as_string)
        content_response = content['response']
        # It's a dictionary - want just the key=listings in the dictionary
        listings = content_response['listings']
        # This is a list of dictionaries
        # Convert this to a pandas data frame
        listings = pd.DataFrame(listings)
        # Append to data frame if there is more than one web page
        if i==1:
            homes = listings
        else:
            homes = homes.append(listings)

    # Keep selected columns
    if homes.empty:
        homes = homes
    else:
        homes = homes[['bathroom_number','bedroom_number','car_spaces','construction_year','datasource_name','guid',
                       'keywords','latitude','longitude','lister_name','listing_type','location_accuracy','price',
                       'property_type','summary','title','updated_in_days']]

    return homes

homes = pd.DataFrame()
postcode_districts = ['PO1','PO2','PO3','PO4','PO5','PO6','PO7','PO8','PO9','PO10','PO11','PO12',
                      'PO13','PO14','PO15','PO16','PO17','PO18','PO19','PO20','PO21','PO22','PO30',
                      'PO31','PO32','PO33','PO34','PO35','PO36','PO37','PO38','PO39','PO40','PO41']
for i in postcode_districts:
    area_name = i
    temp = get_nestoria('rent')
    if i == 0:
        homes = temp
    else:
        homes = homes.append(temp)

print(len(homes))
print(homes.head())
