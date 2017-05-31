# Housing websites

## Overview

Project to collect data about properties for sale or for rent, then use the data to identify caravan homes.

This project collects data from the [Nestoria](http://www.nestoria.co.uk/) and [Zoopla](http://www.zoopla.co.uk/) APIs about properties for sale and rent. It then uses the data collected from Zoopla to build an algorithm which identifies caravan homes with good accuracy.

This work was driven by the need to accurately identify the location of caravan homes for the 2021 Census in England and Wales, as other data sources either record caravan homes inconsistently or not at all.

## How do I use this project?

### Requirements/Pre-requisites

To obtain data from the Zoopla API, an API key is required which can be obtained from the [Zoopla developer pages](http://developer.zoopla.com/). No API key is required for accessing the [Nestoria API](http://www.nestoria.co.uk/help/api).

All code is written using Python 3.4, with different packages required by each file. Overall the following packages are required:
- scikit-learn
- scipy
- numpy
- pandas
- requests

### Data

The Zoopla data collected and used in *Zoopla ML caravans.py* is not shown as it was collected using a specific API key. However example data is shown here:

```
{
  "agent_name": "Fine & Country",
  "county": "Hampshire",
  "description": "Ideal First Time Buy Or Buy To Let Investment • Two Bedrooms • Living Room • Fitted 18ft Kitchen/Dining Room • Family Bathroom",
  "displayable_address": "Acacia Avenue, Fareham",
  "latitude": 50.9,
  "listing_id": 12345678,
  "listing_status": "sale",
  "longitude": -1.1,
  "new_home": "TRUE",
  "num_bathrooms": 2,
  "num_bedrooms": 3,
  "price": 220000,
  "property_type": "Detached house",
  "street_name": "Acacia Avenue Fareham",
  "parkhome": 0
}
```

## Contributors

[Karen Gask](https://github.com/gaskyk), working for the [Office for National Statistics Big Data project](https://www.ons.gov.uk/aboutus/whatwedo/programmesandprojects/theonsbigdataproject)

## Licence

Released under the LICENSE
