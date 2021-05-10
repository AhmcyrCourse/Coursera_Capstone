#  *The best locations where it pays to open restaurant in New York*

## Introduction/Business Problem

Restaurants are one of the most profitable sectors. However, according to one study, 60 percent close or change owners within the first year of operation, 80 percent fail within five years. Usually, restaurants fail with combination of problems that eventually lead to their closure. A bad location is one of the biggest reasons for restaurant failure. For example, a restaurant can sell the best "burger" in the world. If it is in a poor location (hidden, sparsely inhabited, blind and difficult to access) it will have to put in much more effort to fetch customers than to serve them.

In this context, how to define the best locations where it pays to open a restaurant?

Our objective is to recommend the best locations in New York city (well inhabited, close to subways, distant from existing restaurants) to open restaurant. We don’t distinguish the kind of restaurant.

The purpose of this whole exercise is for submission of the final capstone project for the "IBM Data Science" course on Coursera as well as to showcase my data science skills in the real-world application.



## Project Data Source

The data set required for this project provided from four different data sources: 

- Cordinates of the boundaries of Neighborhood Tabulation Areas (NTA) in New York from https://www1.nyc.gov/site/planning/data-maps/open-data/dwn-nynta.page

- Population Numbers By New York City Neighborhood Tabulation Areas (NTA) from https://data.cityofnewyork.us. The link of the CSV file https://query.data.world/s/zdkpdxvomgauu4r3jvymhy57mwtolg

- Location data of New York city subway station from https://data.cityofnewyork.us. The link of the CSV file is https://query.data.world/s/rttrjnk7raatdri6ialljpsucvbv5b
It will help to determine the minimal distance from a NTA to a subway station and the number of subways located in a given radius.

- Location data of restaurants provided from Foursquare API. It will help to determine the minimal distance from a restaurant to a NTA and the number of restaurants located in a given radius.

These data required high pre-processing in order to convert it to a working set, capable of handling machine learning algorithms and visualization operations that were implemented on it.

So, we generate a dataframe with a number of rows corresponding to NTA and columns are:
* longitude and latitude
* population
* minimal distance from a neighborhood location to a subway station
* number of subways located in a given radius
* minimal distance from a restaurant to a neighborhood location
* number of restaurants located in a given radius


The best locations are those where there is no or few restaurants, close to subway stations and well inhabited. 
NTA boundaries and their associated names may not definitively represent neighborhoods. We consider the center of NTA as districts in this exercise.







```python
import wget
import pandas as pd
import json
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans
import folium # map rendering librar
from bs4 import BeautifulSoup # Library to scrape website

import geopandas as gpd

print('Libraries imported.')
```

    Libraries imported.
    

# Data visualization and pre-processing

## Lets load Population Numbers By New York City Neighborhood Tabulation Areas (NTA) dataset in a CSV file
- NTA Name: The name of Neighborhood Tabulation Areas. 
- Population: Population number

We create "df_pop" dataframe


```python
df_pop = pd.read_csv('https://query.data.world/s/zdkpdxvomgauu4r3jvymhy57mwtolg')
df_pop = df_pop[df_pop["Year"]==2010].reset_index(drop = True)
df_pop.rename(columns = {'NTA Code':'DistrictCode'}, inplace =True)
df_pop.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Year</th>
      <th>FIPS County Code</th>
      <th>DistrictCode</th>
      <th>NTA Name</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bronx</td>
      <td>2010</td>
      <td>5</td>
      <td>BX01</td>
      <td>Claremont-Bathgate</td>
      <td>31078</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bronx</td>
      <td>2010</td>
      <td>5</td>
      <td>BX03</td>
      <td>Eastchester-Edenwald-Baychester</td>
      <td>34517</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bronx</td>
      <td>2010</td>
      <td>5</td>
      <td>BX05</td>
      <td>Bedford Park-Fordham North</td>
      <td>54415</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bronx</td>
      <td>2010</td>
      <td>5</td>
      <td>BX06</td>
      <td>Belmont</td>
      <td>27378</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bronx</td>
      <td>2010</td>
      <td>5</td>
      <td>BX07</td>
      <td>Bronxdale</td>
      <td>35538</td>
    </tr>
  </tbody>
</table>
</div>



As we need the coordinates of NTA, we download them and saved in a GeoJson format


```python
filename2 = wget.download("https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/ArcGIS/rest/services/NYC_Neighborhood_Tabulation_Areas/FeatureServer/0/query?where=1=1&outFields=*&outSR=4326&f=pgeojson")
```

    100% [..........................................................................] 6206903 / 6206903


```python
with open(filename2) as json_data:
    newyork_puma = json.load(json_data)
```

## NTA are polygons or MultiPolygons. Our strategy is to determine the centeroid of these features which will be named "District"

- A function that determines the coordinates of the Point representing the center of polygon or multipolygon


```python
def centeroid(arr):
    length = len(arr)
    sum_x = sum([r[0] for r in arr])
    sum_y =  sum([r[1] for r in arr])
    return sum_x/length, sum_y/length
```

Based on the GeoJson file, we generate "District" dataframe that contains:
- Borough
- DistrictName: The name of the NTA 
- DistrictCode: The code of the NTA
- Latitude and Longitude of the centers of NTA


```python
districts_data = newyork_puma['features']
# define the dataframe columns
column_names = ['Borough', 'DistrictCode', 'DistrictName', 'Latitude', 'Longitude'] 

# instantiate the dataframe
districts = pd.DataFrame(columns=column_names)
n=1
for data in districts_data:
    borough = data['properties']['BoroName'] 
    DistrictCode = data['properties']['NTACode']       
    DistrictName = data['properties']['NTAName']
    d = data["geometry"]["coordinates"]
    if len(d)==1: 
        Longitude, Latitude = np.mean(np.array(d[0]), axis=0)
    if len(d)>1 and data["geometry"]["type"]=="Polygon":
        z = [centeroid(f) for f in d]
        Longitude, Latitude  = np.mean(np.array(z), axis=0)
    if len(d)>1 and data["geometry"]["type"]=="MultiPolygon":
        z = [centeroid(f) for f in d[0]]
        Longitude, Latitude  = np.mean(np.array(z), axis=0)
    
    districts = districts.append({'Borough': borough,
                                          'DistrictCode': DistrictCode,
                                          'DistrictName': DistrictName,
                                          'Latitude': Latitude,
                                          'Longitude': Longitude}, ignore_index=True)
districts.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>DistrictCode</th>
      <th>DistrictName</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brooklyn</td>
      <td>BK88</td>
      <td>Borough Park</td>
      <td>40.630667</td>
      <td>-73.987897</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Queens</td>
      <td>QN51</td>
      <td>Murray Hill</td>
      <td>40.768102</td>
      <td>-73.807672</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Queens</td>
      <td>QN27</td>
      <td>East Elmhurst</td>
      <td>40.763467</td>
      <td>-73.866047</td>
    </tr>
  </tbody>
</table>
</div>



We merge the Districts dataframe with df_pop in a new dataframe "df"


```python
df=pd.merge(districts, df_pop, on='DistrictCode')
```


```python
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough_x</th>
      <th>DistrictCode</th>
      <th>DistrictName</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Borough_y</th>
      <th>Year</th>
      <th>FIPS County Code</th>
      <th>NTA Name</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brooklyn</td>
      <td>BK88</td>
      <td>Borough Park</td>
      <td>40.630667</td>
      <td>-73.987897</td>
      <td>Brooklyn</td>
      <td>2010</td>
      <td>47</td>
      <td>Borough Park</td>
      <td>106357</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Queens</td>
      <td>QN51</td>
      <td>Murray Hill</td>
      <td>40.768102</td>
      <td>-73.807672</td>
      <td>Queens</td>
      <td>2010</td>
      <td>81</td>
      <td>Murray Hill</td>
      <td>51739</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Queens</td>
      <td>QN27</td>
      <td>East Elmhurst</td>
      <td>40.763467</td>
      <td>-73.866047</td>
      <td>Queens</td>
      <td>2010</td>
      <td>81</td>
      <td>East Elmhurst</td>
      <td>23150</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Queens</td>
      <td>QN07</td>
      <td>Hollis</td>
      <td>40.710505</td>
      <td>-73.764068</td>
      <td>Queens</td>
      <td>2010</td>
      <td>81</td>
      <td>Hollis</td>
      <td>20269</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brooklyn</td>
      <td>BK25</td>
      <td>Homecrest</td>
      <td>40.598383</td>
      <td>-73.964717</td>
      <td>Brooklyn</td>
      <td>2010</td>
      <td>47</td>
      <td>Homecrest</td>
      <td>44316</td>
    </tr>
  </tbody>
</table>
</div>



We drop the non required columns


```python
df.drop(['Borough_y','Year', 'NTA Name', 'FIPS County Code'],inplace=True,axis=1)

```


```python
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough_x</th>
      <th>DistrictCode</th>
      <th>DistrictName</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brooklyn</td>
      <td>BK88</td>
      <td>Borough Park</td>
      <td>40.630667</td>
      <td>-73.987897</td>
      <td>106357</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Queens</td>
      <td>QN51</td>
      <td>Murray Hill</td>
      <td>40.768102</td>
      <td>-73.807672</td>
      <td>51739</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Queens</td>
      <td>QN27</td>
      <td>East Elmhurst</td>
      <td>40.763467</td>
      <td>-73.866047</td>
      <td>23150</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Queens</td>
      <td>QN07</td>
      <td>Hollis</td>
      <td>40.710505</td>
      <td>-73.764068</td>
      <td>20269</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brooklyn</td>
      <td>BK25</td>
      <td>Homecrest</td>
      <td>40.598383</td>
      <td>-73.964717</td>
      <td>44316</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.rename(columns = {'Borough_x':'Borough'}, inplace =True)
```


```python
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>DistrictCode</th>
      <th>DistrictName</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brooklyn</td>
      <td>BK88</td>
      <td>Borough Park</td>
      <td>40.630667</td>
      <td>-73.987897</td>
      <td>106357</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Queens</td>
      <td>QN51</td>
      <td>Murray Hill</td>
      <td>40.768102</td>
      <td>-73.807672</td>
      <td>51739</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Queens</td>
      <td>QN27</td>
      <td>East Elmhurst</td>
      <td>40.763467</td>
      <td>-73.866047</td>
      <td>23150</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Queens</td>
      <td>QN07</td>
      <td>Hollis</td>
      <td>40.710505</td>
      <td>-73.764068</td>
      <td>20269</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brooklyn</td>
      <td>BK25</td>
      <td>Homecrest</td>
      <td>40.598383</td>
      <td>-73.964717</td>
      <td>44316</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('The dataframe has {} boroughs and {} districts.'.format(
        len(df['Borough'].unique()),
        df.shape[0]
    )
)
```

    The dataframe has 5 boroughs and 195 districts.
    

# Visualization of New York NTA through the coordinates of their calculated centers


```python
address = 'New York City, NY'
geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of New York City are {}, {}.'.format(latitude, longitude))
```

    The geograpical coordinate of New York City are 40.7127281, -74.0060152.
    


```python
# create map of New York using latitude and longitude values
map_newyork = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, district in zip(df['Latitude'], df['Longitude'], df['Borough'], df['DistrictName']):
    label = '{}, {}'.format(district, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_newyork)  
    
map_newyork
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=%3C%21DOCTYPE%20html%3E%0A%3Chead%3E%20%20%20%20%0A%20%20%20%20%3Cmeta%20http-equiv%3D%22content-type%22%20content%3D%22text/html%3B%20charset%3DUTF-8%22%20/%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%3Cscript%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20L_NO_TOUCH%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20L_DISABLE_3D%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%3C/script%3E%0A%20%20%20%20%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//code.jquery.com/jquery-1.12.4.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js%22%3E%3C/script%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//rawcdn.githack.com/python-visualization/folium/master/folium/templates/leaflet.awesome.rotate.css%22/%3E%0A%20%20%20%20%3Cstyle%3Ehtml%2C%20body%20%7Bwidth%3A%20100%25%3Bheight%3A%20100%25%3Bmargin%3A%200%3Bpadding%3A%200%3B%7D%3C/style%3E%0A%20%20%20%20%3Cstyle%3E%23map%20%7Bposition%3Aabsolute%3Btop%3A0%3Bbottom%3A0%3Bright%3A0%3Bleft%3A0%3B%7D%3C/style%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cmeta%20name%3D%22viewport%22%20content%3D%22width%3Ddevice-width%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20initial-scale%3D1.0%2C%20maximum-scale%3D1.0%2C%20user-scalable%3Dno%22%20/%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cstyle%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%23map_66672434f13e4454b4cf37ec34df5800%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20position%3A%20relative%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20width%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20height%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20left%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20top%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%3C/style%3E%0A%20%20%20%20%20%20%20%20%0A%3C/head%3E%0A%3Cbody%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cdiv%20class%3D%22folium-map%22%20id%3D%22map_66672434f13e4454b4cf37ec34df5800%22%20%3E%3C/div%3E%0A%20%20%20%20%20%20%20%20%0A%3C/body%3E%0A%3Cscript%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20map_66672434f13e4454b4cf37ec34df5800%20%3D%20L.map%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22map_66672434f13e4454b4cf37ec34df5800%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20center%3A%20%5B40.7127281%2C%20-74.0060152%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20crs%3A%20L.CRS.EPSG3857%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoom%3A%2010%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoomControl%3A%20true%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20preferCanvas%3A%20false%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20tile_layer_e02a3c91bc3a4004afa35f3289e8ca5e%20%3D%20L.tileLayer%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22https%3A//%7Bs%7D.tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22attribution%22%3A%20%22Data%20by%20%5Cu0026copy%3B%20%5Cu003ca%20href%3D%5C%22http%3A//openstreetmap.org%5C%22%5Cu003eOpenStreetMap%5Cu003c/a%5Cu003e%2C%20under%20%5Cu003ca%20href%3D%5C%22http%3A//www.openstreetmap.org/copyright%5C%22%5Cu003eODbL%5Cu003c/a%5Cu003e.%22%2C%20%22detectRetina%22%3A%20false%2C%20%22maxNativeZoom%22%3A%2018%2C%20%22maxZoom%22%3A%2018%2C%20%22minZoom%22%3A%200%2C%20%22noWrap%22%3A%20false%2C%20%22opacity%22%3A%201%2C%20%22subdomains%22%3A%20%22abc%22%2C%20%22tms%22%3A%20false%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7b250271c21942b09a0d637cbffe69b6%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.63066724072439%2C%20-73.98789731084157%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_7677a36b47e24236b2630799c39f11b6%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_18b22d7e1dd6487399908e298851bd2f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_18b22d7e1dd6487399908e298851bd2f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBorough%20Park%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_7677a36b47e24236b2630799c39f11b6.setContent%28html_18b22d7e1dd6487399908e298851bd2f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7b250271c21942b09a0d637cbffe69b6.bindPopup%28popup_7677a36b47e24236b2630799c39f11b6%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0a18e4f2f8fb4114a5e77ac0f0b3afe5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.76810195255115%2C%20-73.80767235244826%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_bb82bef727ce4e4b9bfa41812ff43fb8%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_64371aeb53ac4cf39fa86915dd4aae4e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_64371aeb53ac4cf39fa86915dd4aae4e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMurray%20Hill%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_bb82bef727ce4e4b9bfa41812ff43fb8.setContent%28html_64371aeb53ac4cf39fa86915dd4aae4e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_0a18e4f2f8fb4114a5e77ac0f0b3afe5.bindPopup%28popup_bb82bef727ce4e4b9bfa41812ff43fb8%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_076a3a7fdec64fa7ab79631062aa6208%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.763466978038146%2C%20-73.86604701197305%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c5a13eebe8a344fb9370e44458814921%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_14922e698f57441ba41f91f049fa89a7%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_14922e698f57441ba41f91f049fa89a7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEast%20Elmhurst%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c5a13eebe8a344fb9370e44458814921.setContent%28html_14922e698f57441ba41f91f049fa89a7%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_076a3a7fdec64fa7ab79631062aa6208.bindPopup%28popup_c5a13eebe8a344fb9370e44458814921%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1b997002cfb140babdea72b04a9d493e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.71050489065737%2C%20-73.76406789189183%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a2ca705a0d9c444cae92d14ed5c82b3d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7aab81451d514762921d544296e48f8f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7aab81451d514762921d544296e48f8f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHollis%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a2ca705a0d9c444cae92d14ed5c82b3d.setContent%28html_7aab81451d514762921d544296e48f8f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_1b997002cfb140babdea72b04a9d493e.bindPopup%28popup_a2ca705a0d9c444cae92d14ed5c82b3d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_df9570aa65a443349772fa747257d123%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.59838251724025%2C%20-73.96471679230402%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_0a0d72ac04fe4b42b13c79ec74a557da%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e86997508a8442d4950059475e19bc2c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e86997508a8442d4950059475e19bc2c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHomecrest%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_0a0d72ac04fe4b42b13c79ec74a557da.setContent%28html_e86997508a8442d4950059475e19bc2c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_df9570aa65a443349772fa747257d123.bindPopup%28popup_0a0d72ac04fe4b42b13c79ec74a557da%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_221282a8de2c48ec9cc9c434069abf64%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.73499121273286%2C%20-73.78552757662317%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_139ebeb66032498ca5607b56412c536e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8fd75e78cc8744a4a9336e76b6e7b801%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8fd75e78cc8744a4a9336e76b6e7b801%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFresh%20Meadows-Utopia%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_139ebeb66032498ca5607b56412c536e.setContent%28html_8fd75e78cc8744a4a9336e76b6e7b801%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_221282a8de2c48ec9cc9c434069abf64.bindPopup%28popup_139ebeb66032498ca5607b56412c536e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_05ff8bcc60b74df9b8c25d1fc5e3b91f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.69272891424838%2C%20-73.76422328361754%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e60901433bc0498ebb3af22e108e0663%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_fd9f61a1c0f24eb08276e4dea608d4bc%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_fd9f61a1c0f24eb08276e4dea608d4bc%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESt.%20Albans%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e60901433bc0498ebb3af22e108e0663.setContent%28html_fd9f61a1c0f24eb08276e4dea608d4bc%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_05ff8bcc60b74df9b8c25d1fc5e3b91f.bindPopup%28popup_e60901433bc0498ebb3af22e108e0663%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ac517c84800e4ccca0ecf4e6a81cc7ad%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.605535727273725%2C%20-73.94637354731918%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_93e36acd1439464a9aaf85621c562257%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_695feff174294a5e91aa1d931fe48bf2%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_695feff174294a5e91aa1d931fe48bf2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMadison%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_93e36acd1439464a9aaf85621c562257.setContent%28html_695feff174294a5e91aa1d931fe48bf2%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ac517c84800e4ccca0ecf4e6a81cc7ad.bindPopup%28popup_93e36acd1439464a9aaf85621c562257%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_65571c7a4bc54e45a41e6dcf7abdc363%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.6407270475426%2C%20-73.97641427573717%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ead9ee6139774d6d940bd6205e5a6942%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_bd77553f6a3b4bb4ba8653a7580e03a0%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_bd77553f6a3b4bb4ba8653a7580e03a0%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EKensington-Ocean%20Parkway%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ead9ee6139774d6d940bd6205e5a6942.setContent%28html_bd77553f6a3b4bb4ba8653a7580e03a0%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_65571c7a4bc54e45a41e6dcf7abdc363.bindPopup%28popup_ead9ee6139774d6d940bd6205e5a6942%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b550391e447e40ef989ff42296d420d6%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.645555965849944%2C%20-73.95203146577258%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_8758ebb6c3f143cda8d820e23bb004c9%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_692e618b09ad4f4baeea996c4b090a55%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_692e618b09ad4f4baeea996c4b090a55%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EErasmus%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_8758ebb6c3f143cda8d820e23bb004c9.setContent%28html_692e618b09ad4f4baeea996c4b090a55%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b550391e447e40ef989ff42296d420d6.bindPopup%28popup_8758ebb6c3f143cda8d820e23bb004c9%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e835050ab9c14f51b17d353f1fd3011f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.81972901702041%2C%20-73.89855941138944%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e387e450781b4ac39cd82fd5e3e66c10%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_19e7704365fb4781a2ca942d1b07a49a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_19e7704365fb4781a2ca942d1b07a49a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELongwood%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e387e450781b4ac39cd82fd5e3e66c10.setContent%28html_19e7704365fb4781a2ca942d1b07a49a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e835050ab9c14f51b17d353f1fd3011f.bindPopup%28popup_e387e450781b4ac39cd82fd5e3e66c10%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e7910db1328b4ee78542cbe2066fdc9f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.69032755636784%2C%20-73.96155179555186%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_8879fcd647a54639897079a84c6b104f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7327f64978a3431db98842a010eb534a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7327f64978a3431db98842a010eb534a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EClinton%20Hill%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_8879fcd647a54639897079a84c6b104f.setContent%28html_7327f64978a3431db98842a010eb534a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e7910db1328b4ee78542cbe2066fdc9f.bindPopup%28popup_8879fcd647a54639897079a84c6b104f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7839594d081e4a728d6c3254dfa0480c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.694748456877164%2C%20-73.73647308003734%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_67fd3314ce154d3a800de053d820e065%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_cd63ac43444c4ed1a516cfc773028ef5%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_cd63ac43444c4ed1a516cfc773028ef5%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECambria%20Heights%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_67fd3314ce154d3a800de053d820e065.setContent%28html_cd63ac43444c4ed1a516cfc773028ef5%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7839594d081e4a728d6c3254dfa0480c.bindPopup%28popup_67fd3314ce154d3a800de053d820e065%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_229c5490b7c542fcbe6f1f700927b06d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.62172356699888%2C%20-73.95822312109114%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c2496ae3faf84738831abe72ef831732%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7c457c7ddeae4fdd8b4ac2ecab281a39%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7c457c7ddeae4fdd8b4ac2ecab281a39%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMidwood%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c2496ae3faf84738831abe72ef831732.setContent%28html_7c457c7ddeae4fdd8b4ac2ecab281a39%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_229c5490b7c542fcbe6f1f700927b06d.bindPopup%28popup_c2496ae3faf84738831abe72ef831732%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ca04982a0ec64bc18a958f37c122a9e9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.58653919698585%2C%20-73.98852108509958%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_94184def689f4c6b93b984106510dac9%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e43c498beee248358483d523424c08c6%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e43c498beee248358483d523424c08c6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGravesend%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_94184def689f4c6b93b984106510dac9.setContent%28html_e43c498beee248358483d523424c08c6%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ca04982a0ec64bc18a958f37c122a9e9.bindPopup%28popup_94184def689f4c6b93b984106510dac9%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_50d5bdbf13a74d6cab307484d467de66%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.59705422913784%2C%20-73.99003771016272%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5a5d733b885f45518b025af9109cdee9%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5e9d9a87709f453c98217184898a9b78%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_5e9d9a87709f453c98217184898a9b78%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBensonhurst%20East%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5a5d733b885f45518b025af9109cdee9.setContent%28html_5e9d9a87709f453c98217184898a9b78%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_50d5bdbf13a74d6cab307484d467de66.bindPopup%28popup_5a5d733b885f45518b025af9109cdee9%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3800096c002149ea98655309e822b4d5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.704028062531066%2C%20-73.95800114307804%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_7841bc2a3cd34b07a8f96ec4529d5ad0%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_58b4def5dc4a435281ba170bbf2b1d2d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_58b4def5dc4a435281ba170bbf2b1d2d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWilliamsburg%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_7841bc2a3cd34b07a8f96ec4529d5ad0.setContent%28html_58b4def5dc4a435281ba170bbf2b1d2d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_3800096c002149ea98655309e822b4d5.bindPopup%28popup_7841bc2a3cd34b07a8f96ec4529d5ad0%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2d93f93be921425d9f8cf1f93e04ccc4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.69242882406202%2C%20-73.95021712857192%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5f9fcd2cf74e409e8d904f006704aa8e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_edd7b37496cb4444869f9a68612ac689%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_edd7b37496cb4444869f9a68612ac689%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBedford%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5f9fcd2cf74e409e8d904f006704aa8e.setContent%28html_edd7b37496cb4444869f9a68612ac689%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2d93f93be921425d9f8cf1f93e04ccc4.bindPopup%28popup_5f9fcd2cf74e409e8d904f006704aa8e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_86354eff5d0f4f4a8ccf1deb50fdc5b4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.68871325334692%2C%20-73.93039003065593%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_302a116eb87a4fa5b79a8bcdc9b480e5%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_353bc4ca25d54b84803de966eb1f954c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_353bc4ca25d54b84803de966eb1f954c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EStuyvesant%20Heights%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_302a116eb87a4fa5b79a8bcdc9b480e5.setContent%28html_353bc4ca25d54b84803de966eb1f954c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_86354eff5d0f4f4a8ccf1deb50fdc5b4.bindPopup%28popup_302a116eb87a4fa5b79a8bcdc9b480e5%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_bd490b2371494430822d5b29092156f1%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.65253253946017%2C%20-73.92138016169179%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e7fdbb216c4a4ae8892dbcb0df1e77a0%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_611288e8cbac4126a1494c71e742c0db%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_611288e8cbac4126a1494c71e742c0db%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERugby-Remsen%20Village%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e7fdbb216c4a4ae8892dbcb0df1e77a0.setContent%28html_611288e8cbac4126a1494c71e742c0db%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_bd490b2371494430822d5b29092156f1.bindPopup%28popup_e7fdbb216c4a4ae8892dbcb0df1e77a0%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_19c2315221f34871bf608a2637908b43%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.86401446372289%2C%20-73.84726540515813%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_8fb1baddcb1b41829ff4aa052e36946c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_92b5b62179304d499aa2967e8b339be1%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_92b5b62179304d499aa2967e8b339be1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EAllerton-Pelham%20Gardens%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_8fb1baddcb1b41829ff4aa052e36946c.setContent%28html_92b5b62179304d499aa2967e8b339be1%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_19c2315221f34871bf608a2637908b43.bindPopup%28popup_8fb1baddcb1b41829ff4aa052e36946c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1d009d2aa1724ec1b896e06b2a9c5ccf%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.64509973465978%2C%20-73.93731980404424%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_9482a26dcfd043e58419a8d664151631%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3f03e83ad1fc49668075a094e1e5dabf%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_3f03e83ad1fc49668075a094e1e5dabf%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEast%20Flatbush-Farragut%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_9482a26dcfd043e58419a8d664151631.setContent%28html_3f03e83ad1fc49668075a094e1e5dabf%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_1d009d2aa1724ec1b896e06b2a9c5ccf.bindPopup%28popup_9482a26dcfd043e58419a8d664151631%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4121c847f9ba48ee90f44eeb952eb964%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.69625842921666%2C%20-73.92543277371934%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6151fc8337414ca4abafd81fa7425575%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e5d1ee00820c43b79f57eefca9e29c2b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e5d1ee00820c43b79f57eefca9e29c2b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBushwick%20South%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6151fc8337414ca4abafd81fa7425575.setContent%28html_e5d1ee00820c43b79f57eefca9e29c2b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4121c847f9ba48ee90f44eeb952eb964.bindPopup%28popup_6151fc8337414ca4abafd81fa7425575%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_673c05b06f394422af7b4e4f46778824%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.69887800952358%2C%20-73.9162909905624%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c5e485f578f9429d941ac8db18a5af24%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7d12b9a903c64eb5b619306e271f6ea9%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7d12b9a903c64eb5b619306e271f6ea9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBushwick%20North%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c5e485f578f9429d941ac8db18a5af24.setContent%28html_7d12b9a903c64eb5b619306e271f6ea9%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_673c05b06f394422af7b4e4f46778824.bindPopup%28popup_c5e485f578f9429d941ac8db18a5af24%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1ba2f068dae841c988d38a9bf2728c4f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.70684231967818%2C%20-73.90362114777328%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_22063c4f207642fd815c31103b72dc05%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_dece579bf0e443c2a899bda4234d1124%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_dece579bf0e443c2a899bda4234d1124%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERidgewood%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_22063c4f207642fd815c31103b72dc05.setContent%28html_dece579bf0e443c2a899bda4234d1124%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_1ba2f068dae841c988d38a9bf2728c4f.bindPopup%28popup_22063c4f207642fd815c31103b72dc05%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_00e81147df3944a3925d8a2c77ce677d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.61784217574173%2C%20-73.97144479747296%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_db00b71f96f1473ea034b3de4f18f679%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_85f1394936014444b978f57dd6a5ef4a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_85f1394936014444b978f57dd6a5ef4a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EOcean%20Parkway%20South%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_db00b71f96f1473ea034b3de4f18f679.setContent%28html_85f1394936014444b978f57dd6a5ef4a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_00e81147df3944a3925d8a2c77ce677d.bindPopup%28popup_db00b71f96f1473ea034b3de4f18f679%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a2fc5120a2114a86b90b1bd97d4dafd3%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.759915157306025%2C%20-73.93726357043789%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c585a7302a494ff4bfaf76bdab4ee839%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9622893f71c8417186f2461a7da095ea%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9622893f71c8417186f2461a7da095ea%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EQueensbridge-Ravenswood-Long%20Island%20City%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c585a7302a494ff4bfaf76bdab4ee839.setContent%28html_9622893f71c8417186f2461a7da095ea%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a2fc5120a2114a86b90b1bd97d4dafd3.bindPopup%28popup_c585a7302a494ff4bfaf76bdab4ee839%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6a0c5e32171c4720ae909b0949ad54e6%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.83492956297838%2C%20-73.87356641456357%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_db848493d7e54bc18fb701cc4f480d4f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_eff49367912740e7be435f346dadf3a2%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_eff49367912740e7be435f346dadf3a2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWest%20Farms-Bronx%20River%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_db848493d7e54bc18fb701cc4f480d4f.setContent%28html_eff49367912740e7be435f346dadf3a2%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_6a0c5e32171c4720ae909b0949ad54e6.bindPopup%28popup_db848493d7e54bc18fb701cc4f480d4f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_bfe97f1696a64538bf92fe16501d3342%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.83709118911526%2C%20-73.85873004522746%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_24bd03817f6541c4a86baba925beb229%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c06c34d7ec95429e85fe8870c0c090b2%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c06c34d7ec95429e85fe8870c0c090b2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EParkchester%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_24bd03817f6541c4a86baba925beb229.setContent%28html_c06c34d7ec95429e85fe8870c0c090b2%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_bfe97f1696a64538bf92fe16501d3342.bindPopup%28popup_24bd03817f6541c4a86baba925beb229%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1a273dcab074490b94e3eccbbf47d75f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.82805780897781%2C%20-73.86797338529935%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b3a4afc07de2492fbd3e4b4def87f06e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7d1d8cf361f94b3497d0a24f33e12bae%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7d1d8cf361f94b3497d0a24f33e12bae%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESoundview-Bruckner%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b3a4afc07de2492fbd3e4b4def87f06e.setContent%28html_7d1d8cf361f94b3497d0a24f33e12bae%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_1a273dcab074490b94e3eccbbf47d75f.bindPopup%28popup_b3a4afc07de2492fbd3e4b4def87f06e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0c70a267ae3a4a44ad42c9f35777c7c2%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.702316036297034%2C%20-73.87663617449986%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_bbc487d5c3cf4386b5e91d6df0c15347%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_857237ae38764f72ae2b0bc47380dcf4%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_857237ae38764f72ae2b0bc47380dcf4%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGlendale%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_bbc487d5c3cf4386b5e91d6df0c15347.setContent%28html_857237ae38764f72ae2b0bc47380dcf4%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_0c70a267ae3a4a44ad42c9f35777c7c2.bindPopup%28popup_bbc487d5c3cf4386b5e91d6df0c15347%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ba1d9676f4a24c9b89e8d4cf32b182be%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.87538107480276%2C%20-73.89601679468754%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f6144572a83b429997188eedf8067324%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_40fcc20cf811450fb5c4197c11c1aa93%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_40fcc20cf811450fb5c4197c11c1aa93%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EVan%20Cortlandt%20Village%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f6144572a83b429997188eedf8067324.setContent%28html_40fcc20cf811450fb5c4197c11c1aa93%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ba1d9676f4a24c9b89e8d4cf32b182be.bindPopup%28popup_f6144572a83b429997188eedf8067324%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5d059c75dfd2404cb12ba01d9363e891%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.7663775596076%2C%20-73.99654837674895%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6736c391e94b4be4907202a590a61982%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_96b3f4e9c90242a1b3a3fcf8ac377ef5%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_96b3f4e9c90242a1b3a3fcf8ac377ef5%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EClinton%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6736c391e94b4be4907202a590a61982.setContent%28html_96b3f4e9c90242a1b3a3fcf8ac377ef5%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_5d059c75dfd2404cb12ba01d9363e891.bindPopup%28popup_6736c391e94b4be4907202a590a61982%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ee29790fc4da4a6a821512c600951d78%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.71679553739745%2C%20-73.74313656380424%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_2b1ee43927704b1aa45da3d41bb1f084%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1a7b3f65d38f49baad904dcc329bcfb9%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1a7b3f65d38f49baad904dcc329bcfb9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EQueens%20Village%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_2b1ee43927704b1aa45da3d41bb1f084.setContent%28html_1a7b3f65d38f49baad904dcc329bcfb9%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ee29790fc4da4a6a821512c600951d78.bindPopup%28popup_2b1ee43927704b1aa45da3d41bb1f084%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d59e33af247543048051749c5d64dc62%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.71881695961429%2C%20-73.88231448905366%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6beb77f907c5416781bdb843e817b7a4%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_944d22b31bf5439895ae40b054d853c3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_944d22b31bf5439895ae40b054d853c3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMiddle%20Village%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6beb77f907c5416781bdb843e817b7a4.setContent%28html_944d22b31bf5439895ae40b054d853c3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_d59e33af247543048051749c5d64dc62.bindPopup%28popup_6beb77f907c5416781bdb843e817b7a4%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_56cdec7a92de4bb09bca77c9b514edd3%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.89988734313774%2C%20-73.85667618402015%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_54d04343492f49968caf3358411b5579%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a70b9385476a40429a561e237fa3c523%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_a70b9385476a40429a561e237fa3c523%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWoodlawn-Wakefield%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_54d04343492f49968caf3358411b5579.setContent%28html_a70b9385476a40429a561e237fa3c523%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_56cdec7a92de4bb09bca77c9b514edd3.bindPopup%28popup_54d04343492f49968caf3358411b5579%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c834cf415eab4739b0e825a31ad08e65%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.86215064493589%2C%20-73.86510060936587%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_baf800dfb3234c388e72e21af64cbd24%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_472794005c8c473cb41567741307e151%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_472794005c8c473cb41567741307e151%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBronxdale%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_baf800dfb3234c388e72e21af64cbd24.setContent%28html_472794005c8c473cb41567741307e151%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c834cf415eab4739b0e825a31ad08e65.bindPopup%28popup_baf800dfb3234c388e72e21af64cbd24%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_02177af6c5e14df4915539a9c6c48e63%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.666864604794505%2C%20-73.89482303145759%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_69df7cef22f645fb9cf371c9d8fe5d6f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d1a33900e8f44e20b82cf41a05a27c47%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d1a33900e8f44e20b82cf41a05a27c47%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEast%20New%20York%20%28Pennsylvania%20Ave%29%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_69df7cef22f645fb9cf371c9d8fe5d6f.setContent%28html_d1a33900e8f44e20b82cf41a05a27c47%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_02177af6c5e14df4915539a9c6c48e63.bindPopup%28popup_69df7cef22f645fb9cf371c9d8fe5d6f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4b95d5b80a73424f90b9ecb06b6ab6fe%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.67643948808078%2C%20-73.81682500213628%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_be0fd71f9091463893ae90f24eafb94f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c2fa6f833dda43adb5c0ae9fb5fd3776%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c2fa6f833dda43adb5c0ae9fb5fd3776%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESouth%20Ozone%20Park%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_be0fd71f9091463893ae90f24eafb94f.setContent%28html_c2fa6f833dda43adb5c0ae9fb5fd3776%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4b95d5b80a73424f90b9ecb06b6ab6fe.bindPopup%28popup_be0fd71f9091463893ae90f24eafb94f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_759bfcbf49fb4079926b08400f7d9e17%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.60242999772926%2C%20-74.00880784889402%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ec095bd9720a41f081bb35eeafbc272c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2a31970a2648429b98235e9a603723c5%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2a31970a2648429b98235e9a603723c5%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBath%20Beach%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ec095bd9720a41f081bb35eeafbc272c.setContent%28html_2a31970a2648429b98235e9a603723c5%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_759bfcbf49fb4079926b08400f7d9e17.bindPopup%28popup_ec095bd9720a41f081bb35eeafbc272c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9d93ab507dab4cb1880d3e592f88cea6%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.60846812608773%2C%20-73.99580961545192%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a711167bf734420bb17a11621ec684de%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b771490df5744d4780d130f9ed62df38%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b771490df5744d4780d130f9ed62df38%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBensonhurst%20West%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a711167bf734420bb17a11621ec684de.setContent%28html_b771490df5744d4780d130f9ed62df38%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9d93ab507dab4cb1880d3e592f88cea6.bindPopup%28popup_a711167bf734420bb17a11621ec684de%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c9cd319ce30541148acc56333bc2e2c8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.72342363848227%2C%20-73.90359992470624%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d0237945c18d4e36917435a4fa137949%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_eef666faf0d144769a551f763944155c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_eef666faf0d144769a551f763944155c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMaspeth%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d0237945c18d4e36917435a4fa137949.setContent%28html_eef666faf0d144769a551f763944155c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c9cd319ce30541148acc56333bc2e2c8.bindPopup%28popup_d0237945c18d4e36917435a4fa137949%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fc17c3790b564e17b1f8c0f32bc96e7e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.78701325198355%2C%20-73.86831872142966%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a0f71463ca8e4b509e74dda6aa134c49%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ecc5464fdf3248ddb563f18f44378fa1%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ecc5464fdf3248ddb563f18f44378fa1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECollege%20Point%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a0f71463ca8e4b509e74dda6aa134c49.setContent%28html_ecc5464fdf3248ddb563f18f44378fa1%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_fc17c3790b564e17b1f8c0f32bc96e7e.bindPopup%28popup_a0f71463ca8e4b509e74dda6aa134c49%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a5c174b07be84016bf003ee806c24270%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.793610654519455%2C%20-73.88075017772819%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e2db6642aab34990b7f8257cf5bd2c66%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e37eafc0f03f4344825d1ac22e6be01b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e37eafc0f03f4344825d1ac22e6be01b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERikers%20Island%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e2db6642aab34990b7f8257cf5bd2c66.setContent%28html_e37eafc0f03f4344825d1ac22e6be01b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a5c174b07be84016bf003ee806c24270.bindPopup%28popup_e2db6642aab34990b7f8257cf5bd2c66%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b5fe8ef95f734475ab53c1c1f3317748%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.556929376599584%2C%20-74.1897745335594%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_237ee7655ab84bd3b4379ba051856f1a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_aba1101b61fb406d9678a421b30441da%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_aba1101b61fb406d9678a421b30441da%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EArden%20Heights%2C%20Staten%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_237ee7655ab84bd3b4379ba051856f1a.setContent%28html_aba1101b61fb406d9678a421b30441da%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b5fe8ef95f734475ab53c1c1f3317748.bindPopup%28popup_237ee7655ab84bd3b4379ba051856f1a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9613c080f1d743818881659d9fd875de%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.83511585909493%2C%20-73.83958471919684%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3890098494b3477bbf34df5f2c877e30%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9ea8c9e402a64766ac4e2eb7f50c54ec%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9ea8c9e402a64766ac4e2eb7f50c54ec%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESchuylerville-Throgs%20Neck-Edgewater%20Park%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3890098494b3477bbf34df5f2c877e30.setContent%28html_9ea8c9e402a64766ac4e2eb7f50c54ec%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9613c080f1d743818881659d9fd875de.bindPopup%28popup_3890098494b3477bbf34df5f2c877e30%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_61441ce48f5e48cc8ed386564d0ab13f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.520513743142715%2C%20-74.19289355142742%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_51ced90438c14ce8ad4accbff16cb075%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d61ade2186dc449c99efd6f4a005372a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d61ade2186dc449c99efd6f4a005372a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EAnnadale-Huguenot-Prince%26%2339%3Bs%20Bay-Eltingville%2C%20Staten%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_51ced90438c14ce8ad4accbff16cb075.setContent%28html_d61ade2186dc449c99efd6f4a005372a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_61441ce48f5e48cc8ed386564d0ab13f.bindPopup%28popup_51ced90438c14ce8ad4accbff16cb075%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ce3064ca3e804d6b856e7bb095c0a625%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.69473916696892%2C%20-74.00144418241418%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_74d48093c05c4cb491b7aa5a237ca81c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6a15ee7c1abf4e95a061249d29878e0c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_6a15ee7c1abf4e95a061249d29878e0c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBattery%20Park%20City-Lower%20Manhattan%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_74d48093c05c4cb491b7aa5a237ca81c.setContent%28html_6a15ee7c1abf4e95a061249d29878e0c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ce3064ca3e804d6b856e7bb095c0a625.bindPopup%28popup_74d48093c05c4cb491b7aa5a237ca81c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4307cf375ef3438a90bc2bb66a3bcb94%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.77523608214009%2C%20-73.98836366766845%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e46a074975d34cf2a15316e5fd6743e2%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_084695636f0542f38c37d9b701a71a94%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_084695636f0542f38c37d9b701a71a94%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELincoln%20Square%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e46a074975d34cf2a15316e5fd6743e2.setContent%28html_084695636f0542f38c37d9b701a71a94%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4307cf375ef3438a90bc2bb66a3bcb94.bindPopup%28popup_e46a074975d34cf2a15316e5fd6743e2%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6f6fb45e75b54b8294c0a54d892af9d9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.75675923700698%2C%20-73.98285820774203%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_9e1d7fcf68394594866ebf5a0e333492%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b09d5963b3ad4857b9c25db2de4bc4cc%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b09d5963b3ad4857b9c25db2de4bc4cc%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMidtown-Midtown%20South%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_9e1d7fcf68394594866ebf5a0e333492.setContent%28html_b09d5963b3ad4857b9c25db2de4bc4cc%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_6f6fb45e75b54b8294c0a54d892af9d9.bindPopup%28popup_9e1d7fcf68394594866ebf5a0e333492%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e1315dfd199d4de98863f9eecd131100%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.77528339839104%2C%20-73.96081636591553%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_560dbe8bcf4342c1811ea1f72bc74405%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b6d390fc129e44adbcfc324c6cf67e28%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b6d390fc129e44adbcfc324c6cf67e28%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EUpper%20East%20Side-Carnegie%20Hill%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_560dbe8bcf4342c1811ea1f72bc74405.setContent%28html_b6d390fc129e44adbcfc324c6cf67e28%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e1315dfd199d4de98863f9eecd131100.bindPopup%28popup_560dbe8bcf4342c1811ea1f72bc74405%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9a4f6471e8314b4485941883d2136819%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.83816913222687%2C%20-73.92568279872012%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c1d3c001d08e41eba3fc4893b148ed82%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9c741c691b39482b965ad5805f4e3671%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9c741c691b39482b965ad5805f4e3671%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHighbridge%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c1d3c001d08e41eba3fc4893b148ed82.setContent%28html_9c741c691b39482b965ad5805f4e3671%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9a4f6471e8314b4485941883d2136819.bindPopup%28popup_c1d3c001d08e41eba3fc4893b148ed82%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3b33171ce1c94435afbd0d7df80cf3f6%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.57558555029789%2C%20-73.98680478818761%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_57a9da539e0143a3bcf20b482b06621c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8a76c57481724a0e8bd8970837049278%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8a76c57481724a0e8bd8970837049278%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESeagate-Coney%20Island%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_57a9da539e0143a3bcf20b482b06621c.setContent%28html_8a76c57481724a0e8bd8970837049278%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_3b33171ce1c94435afbd0d7df80cf3f6.bindPopup%28popup_57a9da539e0143a3bcf20b482b06621c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e569565a86204f1890742d5e6720121f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.89867392496872%2C%20-73.90813596446601%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_35152b73128744b78f8c24c8dcec6744%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ac7ba858440046009fabf40a8e9abb60%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ac7ba858440046009fabf40a8e9abb60%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENorth%20Riverdale-Fieldston-Riverdale%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_35152b73128744b78f8c24c8dcec6744.setContent%28html_ac7ba858440046009fabf40a8e9abb60%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e569565a86204f1890742d5e6720121f.bindPopup%28popup_35152b73128744b78f8c24c8dcec6744%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_db5675c3a25f44fcb9b87054c8ff28b7%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.88246427444345%2C%20-73.91168466082128%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_8d6766b089914cd6adffd868b4958000%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_230d18fc579f44cc8220a43a8cf5ba15%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_230d18fc579f44cc8220a43a8cf5ba15%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESpuyten%20Duyvil-Kingsbridge%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_8d6766b089914cd6adffd868b4958000.setContent%28html_230d18fc579f44cc8220a43a8cf5ba15%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_db5675c3a25f44fcb9b87054c8ff28b7.bindPopup%28popup_8d6766b089914cd6adffd868b4958000%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_94b78d4c16cd422a844ec72337547b95%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.655330720701166%2C%20-73.97751902838327%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b4de51e89e6a410eae532065600bfdab%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4360e9077cf244e9b4b7c1a29fa7843f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_4360e9077cf244e9b4b7c1a29fa7843f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWindsor%20Terrace%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b4de51e89e6a410eae532065600bfdab.setContent%28html_4360e9077cf244e9b4b7c1a29fa7843f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_94b78d4c16cd422a844ec72337547b95.bindPopup%28popup_b4de51e89e6a410eae532065600bfdab%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6f44cf7a9ca7462885b0a1f00807bc53%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.646783577767835%2C%20-73.88844556625725%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_fbdc85c2deb84cad937b2a4f6d43911c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_978a78ad58f04023b6e1fa66ae5ff94a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_978a78ad58f04023b6e1fa66ae5ff94a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECanarsie%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_fbdc85c2deb84cad937b2a4f6d43911c.setContent%28html_978a78ad58f04023b6e1fa66ae5ff94a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_6f44cf7a9ca7462885b0a1f00807bc53.bindPopup%28popup_fbdc85c2deb84cad937b2a4f6d43911c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_31e3115eea614e70aadcd11c8fc0c8a6%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.637150615011414%2C%20-73.96218748121711%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ec7e0a2e938c412bb2f23391f2e5b55f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_327d09535fa243e799e496d5455920e6%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_327d09535fa243e799e496d5455920e6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFlatbush%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ec7e0a2e938c412bb2f23391f2e5b55f.setContent%28html_327d09535fa243e799e496d5455920e6%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_31e3115eea614e70aadcd11c8fc0c8a6.bindPopup%28popup_ec7e0a2e938c412bb2f23391f2e5b55f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6705f33df1e749bd8156368c9144e6b5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.628143803535785%2C%20-73.92925474029212%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e11ab5e9512b40f886eeaae6d0ce7540%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_069272814db64dc09ecba7261c2bf317%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_069272814db64dc09ecba7261c2bf317%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFlatlands%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e11ab5e9512b40f886eeaae6d0ce7540.setContent%28html_069272814db64dc09ecba7261c2bf317%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_6705f33df1e749bd8156368c9144e6b5.bindPopup%28popup_e11ab5e9512b40f886eeaae6d0ce7540%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d78faa231abc4609a9d0481fbca8f257%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.76622160095584%2C%20-73.74800208387943%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1c98f1f6760d426bb7fc658fee15593d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4e5524a6fdb04c77b631650ad2c2c13e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_4e5524a6fdb04c77b631650ad2c2c13e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDouglas%20Manor-Douglaston-Little%20Neck%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1c98f1f6760d426bb7fc658fee15593d.setContent%28html_4e5524a6fdb04c77b631650ad2c2c13e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_d78faa231abc4609a9d0481fbca8f257.bindPopup%28popup_1c98f1f6760d426bb7fc658fee15593d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_776d0b429902420193538e2420becc38%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.82200854267259%2C%20-73.8782656420284%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a6d1bbcbf8854d4d965b86b604962197%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_91675875f6714df7b05af034c4312912%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_91675875f6714df7b05af034c4312912%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESoundview-Castle%20Hill-Clason%20Point-Harding%20Park%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a6d1bbcbf8854d4d965b86b604962197.setContent%28html_91675875f6714df7b05af034c4312912%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_776d0b429902420193538e2420becc38.bindPopup%28popup_a6d1bbcbf8854d4d965b86b604962197%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9826ea8d04324e0e9f9e844584894298%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.67162727402453%2C%20-73.98460145039371%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5435b9f2be6d410983bed5b9fafaaa44%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_67f22c9753c843caa92e4395328bdf7d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_67f22c9753c843caa92e4395328bdf7d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EPark%20Slope-Gowanus%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5435b9f2be6d410983bed5b9fafaaa44.setContent%28html_67f22c9753c843caa92e4395328bdf7d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9826ea8d04324e0e9f9e844584894298.bindPopup%28popup_5435b9f2be6d410983bed5b9fafaaa44%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0db2a205493e4e5b94b9043591aa8a1c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.53016056751285%2C%20-74.23697523428403%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_05b3a4cd95de4ed58316de88fc5ecf15%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_cd61844da52c4b3eb0b52ea67c524135%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_cd61844da52c4b3eb0b52ea67c524135%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECharleston-Richmond%20Valley-Tottenville%2C%20Staten%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_05b3a4cd95de4ed58316de88fc5ecf15.setContent%28html_cd61844da52c4b3eb0b52ea67c524135%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_0db2a205493e4e5b94b9043591aa8a1c.bindPopup%28popup_05b3a4cd95de4ed58316de88fc5ecf15%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_78bf8d434add41bfb483fb2a444f2699%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.54069629722725%2C%20-74.20968768742027%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b94dea2062af47eabb2be29f6c7fc4c7%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d230201b7b56471d9bc345768768b760%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d230201b7b56471d9bc345768768b760%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERossville-Woodrow%2C%20Staten%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b94dea2062af47eabb2be29f6c7fc4c7.setContent%28html_d230201b7b56471d9bc345768768b760%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_78bf8d434add41bfb483fb2a444f2699.bindPopup%28popup_b94dea2062af47eabb2be29f6c7fc4c7%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7dceae73b6ca478e9292dc1bf88b970e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.78050399554715%2C%20-73.93852233514457%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_8fbc4d0b92f04953927e14884e9a1a87%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5cb3e64f30374ee5bd35042ad21e1cbd%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_5cb3e64f30374ee5bd35042ad21e1cbd%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EYorkville%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_8fbc4d0b92f04953927e14884e9a1a87.setContent%28html_5cb3e64f30374ee5bd35042ad21e1cbd%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7dceae73b6ca478e9292dc1bf88b970e.bindPopup%28popup_8fbc4d0b92f04953927e14884e9a1a87%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_93a5deaab8f54d5f9f1e4bad30db4e73%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.750222996217644%2C%20-73.79027120508972%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_337c17c8682143a0bd390fcd945b37bc%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7a4aa87356814eefa42ebd784540224e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7a4aa87356814eefa42ebd784540224e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EAuburndale%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_337c17c8682143a0bd390fcd945b37bc.setContent%28html_7a4aa87356814eefa42ebd784540224e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_93a5deaab8f54d5f9f1e4bad30db4e73.bindPopup%28popup_337c17c8682143a0bd390fcd945b37bc%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1def6628546f4fc1ba409d89261bd00d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.7529028355838%2C%20-73.8065523753513%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f0d97cd054c347579d98c555cf35776f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_db33637f03df46bbb25749a076399da1%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_db33637f03df46bbb25749a076399da1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEast%20Flushing%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f0d97cd054c347579d98c555cf35776f.setContent%28html_db33637f03df46bbb25749a076399da1%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_1def6628546f4fc1ba409d89261bd00d.bindPopup%28popup_f0d97cd054c347579d98c555cf35776f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_bb927eb863194dfeb9f7b89ecbee8df0%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.82565642729767%2C%20-73.9049875971515%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c44ef0b0cea04e90be26b68b6d3b28ef%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e6d7f859708847e48e0ff06e981b4479%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e6d7f859708847e48e0ff06e981b4479%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMorrisania-Melrose%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c44ef0b0cea04e90be26b68b6d3b28ef.setContent%28html_e6d7f859708847e48e0ff06e981b4479%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_bb927eb863194dfeb9f7b89ecbee8df0.bindPopup%28popup_c44ef0b0cea04e90be26b68b6d3b28ef%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9d520b4d110a44f6b866e73cc6ee938a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.85601276666469%2C%20-73.92208086039953%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4cc19e3ed55d421cb0a2e2e8af00576d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_dd817b26a26a48d0b36e94e2d14cb1e2%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_dd817b26a26a48d0b36e94e2d14cb1e2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMarble%20Hill-Inwood%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4cc19e3ed55d421cb0a2e2e8af00576d.setContent%28html_dd817b26a26a48d0b36e94e2d14cb1e2%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9d520b4d110a44f6b866e73cc6ee938a.bindPopup%28popup_4cc19e3ed55d421cb0a2e2e8af00576d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b7b6db4c273c43d4ace2c607e77b0b2b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.74911729284833%2C%20-74.00593919513614%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4bddc396b674458699a6d2b39428834a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a8866598a90542c09ef9a5fb7a631e9c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_a8866598a90542c09ef9a5fb7a631e9c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHudson%20Yards-Chelsea-Flatiron-Union%20Square%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4bddc396b674458699a6d2b39428834a.setContent%28html_a8866598a90542c09ef9a5fb7a631e9c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b7b6db4c273c43d4ace2c607e77b0b2b.bindPopup%28popup_4bddc396b674458699a6d2b39428834a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f7ba74e7a8a546cf87014d3dba0f4a66%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.734565184788664%2C%20-74.00621066209094%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_dbe8ccbb837548508239a81ec7ce94b6%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_88e6edea3aff4927abede8cf71113e9e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_88e6edea3aff4927abede8cf71113e9e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWest%20Village%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_dbe8ccbb837548508239a81ec7ce94b6.setContent%28html_88e6edea3aff4927abede8cf71113e9e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f7ba74e7a8a546cf87014d3dba0f4a66.bindPopup%28popup_dbe8ccbb837548508239a81ec7ce94b6%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_00d28b9d34cf477a8ede87f66433bb16%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.79083355054379%2C%20-73.97692336107843%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5e1cfa47363f448e9a6d172e16b856ba%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c8c7d0488fcf4e96bde3ce46cc3e235a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c8c7d0488fcf4e96bde3ce46cc3e235a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EUpper%20West%20Side%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5e1cfa47363f448e9a6d172e16b856ba.setContent%28html_c8c7d0488fcf4e96bde3ce46cc3e235a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_00d28b9d34cf477a8ede87f66433bb16.bindPopup%28popup_5e1cfa47363f448e9a6d172e16b856ba%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7f4d974ea045485eb81847c18d999f62%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.878386920613494%2C%20-73.88204502249278%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_08b3d18c2bdb43de80be7b8a8ec91262%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1497943dce9a41f7b8a12c3a4570a99a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1497943dce9a41f7b8a12c3a4570a99a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENorwood%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_08b3d18c2bdb43de80be7b8a8ec91262.setContent%28html_1497943dce9a41f7b8a12c3a4570a99a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7f4d974ea045485eb81847c18d999f62.bindPopup%28popup_08b3d18c2bdb43de80be7b8a8ec91262%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_573566895f444c29bf45e28ec8e63e4e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.86895485131807%2C%20-73.88923746355657%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b7c68dc85a254447b7b61341005b4429%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_91abcf34a42447dda877ebd9ff0846c1%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_91abcf34a42447dda877ebd9ff0846c1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBedford%20Park-Fordham%20North%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b7c68dc85a254447b7b61341005b4429.setContent%28html_91abcf34a42447dda877ebd9ff0846c1%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_573566895f444c29bf45e28ec8e63e4e.bindPopup%28popup_b7c68dc85a254447b7b61341005b4429%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d56e11fcdf7a4a40af42024bc8a5d5ce%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.85862025289771%2C%20-73.89898336594474%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b9dedd9dfaef4322a551ecfcead9050f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_cca28b100d6145bf8283d5d28337135f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_cca28b100d6145bf8283d5d28337135f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFordham%20South%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b9dedd9dfaef4322a551ecfcead9050f.setContent%28html_cca28b100d6145bf8283d5d28337135f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_d56e11fcdf7a4a40af42024bc8a5d5ce.bindPopup%28popup_b9dedd9dfaef4322a551ecfcead9050f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e11e8c663fc24d3ab75c74731ebb5db0%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.84793765115405%2C%20-73.90629663693664%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4d05a22f618e45f985bd5957737c5fe1%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_96e9311da9f7444eac79acf4ec2a45a9%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_96e9311da9f7444eac79acf4ec2a45a9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMount%20Hope%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4d05a22f618e45f985bd5957737c5fe1.setContent%28html_96e9311da9f7444eac79acf4ec2a45a9%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e11e8c663fc24d3ab75c74731ebb5db0.bindPopup%28popup_4d05a22f618e45f985bd5957737c5fe1%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_69e49590d5454792ba7fc941712280c5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.88153365210217%2C%20-73.83114967645375%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_0026272707654651a858af8646cd8217%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_abb3568ee49f41ba8e90aecd02e249b2%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_abb3568ee49f41ba8e90aecd02e249b2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEastchester-Edenwald-Baychester%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_0026272707654651a858af8646cd8217.setContent%28html_abb3568ee49f41ba8e90aecd02e249b2%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_69e49590d5454792ba7fc941712280c5.bindPopup%28popup_0026272707654651a858af8646cd8217%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_8345330635934678a790ad50c9644d85%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.87554174150015%2C%20-73.83033723021865%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3224bea7749c4efa9b10555302a6a1f4%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7ca693bccb3e4b6fab784bd75dd08a1e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7ca693bccb3e4b6fab784bd75dd08a1e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECo-op%20City%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3224bea7749c4efa9b10555302a6a1f4.setContent%28html_7ca693bccb3e4b6fab784bd75dd08a1e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_8345330635934678a790ad50c9644d85.bindPopup%28popup_3224bea7749c4efa9b10555302a6a1f4%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9845517d213d40879db1c28d231ce1dd%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.587194483030345%2C%20-73.93293132404574%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_dd3731a2364b47c09668205ce7694b9a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_93b7b8978620463fb0a0a110d08ff953%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_93b7b8978620463fb0a0a110d08ff953%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESheepshead%20Bay-Gerritsen%20Beach-Manhattan%20Beach%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_dd3731a2364b47c09668205ce7694b9a.setContent%28html_93b7b8978620463fb0a0a110d08ff953%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9845517d213d40879db1c28d231ce1dd.bindPopup%28popup_dd3731a2364b47c09668205ce7694b9a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_8306034a131d48039a638b0edad26e44%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.61888872005983%2C%20-74.08464767416052%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e84bc57243e346d28cdfdeaddd199717%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3c4f8774edbb4d219d41248b7fa3cc13%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_3c4f8774edbb4d219d41248b7fa3cc13%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGrymes%20Hill-Clifton-Fox%20Hills%2C%20Staten%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e84bc57243e346d28cdfdeaddd199717.setContent%28html_3c4f8774edbb4d219d41248b7fa3cc13%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_8306034a131d48039a638b0edad26e44.bindPopup%28popup_e84bc57243e346d28cdfdeaddd199717%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1e4e94311bb14ab7b94887b8a6447270%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.64392864694343%2C%20-73.74418224190549%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_2e128670345b438c88e52fc1cc84f58b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4fabc52a114d400cbfafc1a2f3a9ebce%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_4fabc52a114d400cbfafc1a2f3a9ebce%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERosedale%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_2e128670345b438c88e52fc1cc84f58b.setContent%28html_4fabc52a114d400cbfafc1a2f3a9ebce%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_1e4e94311bb14ab7b94887b8a6447270.bindPopup%28popup_2e128670345b438c88e52fc1cc84f58b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ae44893e695d4098b21c9b98eecda997%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.56941757899068%2C%20-74.10571515166109%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_97076c734ac846b69ed2dd18ec0a339e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4a01922c69ee4562b8cb54e963a7473b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_4a01922c69ee4562b8cb54e963a7473b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENew%20Dorp-Midland%20Beach%2C%20Staten%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_97076c734ac846b69ed2dd18ec0a339e.setContent%28html_4a01922c69ee4562b8cb54e963a7473b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ae44893e695d4098b21c9b98eecda997.bindPopup%28popup_97076c734ac846b69ed2dd18ec0a339e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6a050128edf946848620da0980f9605d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.69670023609687%2C%20-73.99482050342823%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_7f80dd9bc0db4d898e5350d7173258f5%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b118effa692b4f1e95c8b57481b88497%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b118effa692b4f1e95c8b57481b88497%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBrooklyn%20Heights-Cobble%20Hill%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_7f80dd9bc0db4d898e5350d7173258f5.setContent%28html_b118effa692b4f1e95c8b57481b88497%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_6a050128edf946848620da0980f9605d.bindPopup%28popup_7f80dd9bc0db4d898e5350d7173258f5%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f0e759153ed440568929c0ff23f1a8f9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.61321829793141%2C%20-73.90208076528795%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4116bee9565646fdaa01ba95a266fd85%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6b6349a8645b4b2a958de660f64047b3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_6b6349a8645b4b2a958de660f64047b3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGeorgetown-Marine%20Park-Bergen%20Beach-Mill%20Basin%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4116bee9565646fdaa01ba95a266fd85.setContent%28html_6b6349a8645b4b2a958de660f64047b3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f0e759153ed440568929c0ff23f1a8f9.bindPopup%28popup_4116bee9565646fdaa01ba95a266fd85%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_102c51f803ab49019ebaf759dfab4a66%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.85436955600512%2C%20-73.85320902336122%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ade8a43c951c43c88c30899bfa2f24cb%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_53f7f83f61664dfc8fb08b1f1485e219%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_53f7f83f61664dfc8fb08b1f1485e219%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EPelham%20Parkway%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ade8a43c951c43c88c30899bfa2f24cb.setContent%28html_53f7f83f61664dfc8fb08b1f1485e219%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_102c51f803ab49019ebaf759dfab4a66.bindPopup%28popup_ade8a43c951c43c88c30899bfa2f24cb%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a2ae338476904a91be3a00e7c2e1edc6%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.675952030306874%2C%20-74.01045393977961%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_638c3017c3374d8aafa5ff9005fee097%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_86b020dfaa564bfc82eafc62d3d2dfbe%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_86b020dfaa564bfc82eafc62d3d2dfbe%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECarroll%20Gardens-Columbia%20Street-Red%20Hook%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_638c3017c3374d8aafa5ff9005fee097.setContent%28html_86b020dfaa564bfc82eafc62d3d2dfbe%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a2ae338476904a91be3a00e7c2e1edc6.bindPopup%28popup_638c3017c3374d8aafa5ff9005fee097%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c21ece7c47594978ba803c7ea0a6cebf%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.762940033295195%2C%20-73.76786513389712%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_22b8fa01f88e4184bba2a3ffbaf89707%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_36b172d89ecb43e18cb765d08f511a9a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_36b172d89ecb43e18cb765d08f511a9a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBayside-Bayside%20Hills%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_22b8fa01f88e4184bba2a3ffbaf89707.setContent%28html_36b172d89ecb43e18cb765d08f511a9a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c21ece7c47594978ba803c7ea0a6cebf.bindPopup%28popup_22b8fa01f88e4184bba2a3ffbaf89707%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_70d52623c1424325b1386aacc0c32747%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.827993805267916%2C%20-73.95054568805168%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e928419e7ce54045babffc1bcbf2cc00%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a0f29cd8dd674e798e1f49f0543c354b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_a0f29cd8dd674e798e1f49f0543c354b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHamilton%20Heights%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e928419e7ce54045babffc1bcbf2cc00.setContent%28html_a0f29cd8dd674e798e1f49f0543c354b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_70d52623c1424325b1386aacc0c32747.bindPopup%28popup_e928419e7ce54045babffc1bcbf2cc00%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_37b9e0ef685b44969a484337d301e69f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.78819344298152%2C%20-73.9423712473909%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5958d7e8cb794dda96f28f1288ee8289%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_62ddc62b43224877bfc31f5856135020%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_62ddc62b43224877bfc31f5856135020%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEast%20Harlem%20South%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5958d7e8cb794dda96f28f1288ee8289.setContent%28html_62ddc62b43224877bfc31f5856135020%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_37b9e0ef685b44969a484337d301e69f.bindPopup%28popup_5958d7e8cb794dda96f28f1288ee8289%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_15a3b8763d1d428c8df2b92878ff6364%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.77237525329967%2C%20-73.93029333323034%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a3320306f7964f12a9ac24833f891fc2%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1a6420553a99441d91889beb510714c6%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1a6420553a99441d91889beb510714c6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EOld%20Astoria%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a3320306f7964f12a9ac24833f891fc2.setContent%28html_1a6420553a99441d91889beb510714c6%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_15a3b8763d1d428c8df2b92878ff6364.bindPopup%28popup_a3320306f7964f12a9ac24833f891fc2%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a675d02e16de4c4a8417b28764a37a5e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.68548712038733%2C%20-74.0095551305848%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_9be6154ad9e343da802390e98936bcb9%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6f3ad8c906104ea0a4f78dd74ec58fdf%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_6f3ad8c906104ea0a4f78dd74ec58fdf%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3Epark-cemetery-etc-Manhattan%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_9be6154ad9e343da802390e98936bcb9.setContent%28html_6f3ad8c906104ea0a4f78dd74ec58fdf%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a675d02e16de4c4a8417b28764a37a5e.bindPopup%28popup_9be6154ad9e343da802390e98936bcb9%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f1a6a5f8726e41ec93a6eec9c4d45528%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.78204506778165%2C%20-73.89575953532771%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_fa45747dd95648cdbc56ff50eec5ad51%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6a2bc1eb1e7a4cf8bf2005fd16f0fc8d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_6a2bc1eb1e7a4cf8bf2005fd16f0fc8d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESteinway%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_fa45747dd95648cdbc56ff50eec5ad51.setContent%28html_6a2bc1eb1e7a4cf8bf2005fd16f0fc8d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f1a6a5f8726e41ec93a6eec9c4d45528.bindPopup%28popup_fa45747dd95648cdbc56ff50eec5ad51%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ba68feed77c34991b024b02404b6cd5a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.7878416482504%2C%20-73.78009382140854%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_55b344c05b36467a983e7e5146c73dde%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_204900ecf6ab48fd90f9678d378d0ddf%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_204900ecf6ab48fd90f9678d378d0ddf%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFt.%20Totten-Bay%20Terrace-Clearview%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_55b344c05b36467a983e7e5146c73dde.setContent%28html_204900ecf6ab48fd90f9678d378d0ddf%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ba68feed77c34991b024b02404b6cd5a.bindPopup%28popup_55b344c05b36467a983e7e5146c73dde%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5f7937091ea74f688082d7d6fa5e6ff4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.80347835059119%2C%20-73.95096528365056%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4a6df6d9902f45efbfb6e01a9425e014%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4cb1f7d2da304099b29d1f3fad6d0cca%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_4cb1f7d2da304099b29d1f3fad6d0cca%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECentral%20Harlem%20South%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4a6df6d9902f45efbfb6e01a9425e014.setContent%28html_4cb1f7d2da304099b29d1f3fad6d0cca%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_5f7937091ea74f688082d7d6fa5e6ff4.bindPopup%28popup_4a6df6d9902f45efbfb6e01a9425e014%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_dcad4144a6ff4d0c9a24c2ce3be948ff%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.80515483757544%2C%20-73.93494660371782%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e56ba2bef9f5453e9ade005c077e968c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d54e9277af0d468189dc1ca374b41f24%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d54e9277af0d468189dc1ca374b41f24%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEast%20Harlem%20North%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e56ba2bef9f5453e9ade005c077e968c.setContent%28html_d54e9277af0d468189dc1ca374b41f24%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_dcad4144a6ff4d0c9a24c2ce3be948ff.bindPopup%28popup_e56ba2bef9f5453e9ade005c077e968c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_15a5298c764044b0aa1e9234dbe2c110%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.75983142714458%2C%20-73.82808289762094%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1047e281a8644e87ae794d00d3dc94bc%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_08bd37c0182945288253e564f40d342a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_08bd37c0182945288253e564f40d342a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFlushing%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1047e281a8644e87ae794d00d3dc94bc.setContent%28html_08bd37c0182945288253e564f40d342a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_15a5298c764044b0aa1e9234dbe2c110.bindPopup%28popup_1047e281a8644e87ae794d00d3dc94bc%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_68907605e5a84918a5c73aeb0647b4a4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.639728080410194%2C%20-74.00705812913509%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_929c0508af7248fd833e9ac995a9ee60%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_165d3c228fba48deaf4866400afd5314%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_165d3c228fba48deaf4866400afd5314%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESunset%20Park%20East%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_929c0508af7248fd833e9ac995a9ee60.setContent%28html_165d3c228fba48deaf4866400afd5314%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_68907605e5a84918a5c73aeb0647b4a4.bindPopup%28popup_929c0508af7248fd833e9ac995a9ee60%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d187437a07144f4f8bcc0c96951f0846%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.743679310969874%2C%20-73.75477073705943%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e86dbacafc8e435491dd6ccea04c1034%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_19ab16b1f3fc4ad2a46b95937ce32347%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_19ab16b1f3fc4ad2a46b95937ce32347%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EOakland%20Gardens%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e86dbacafc8e435491dd6ccea04c1034.setContent%28html_19ab16b1f3fc4ad2a46b95937ce32347%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_d187437a07144f4f8bcc0c96951f0846.bindPopup%28popup_e86dbacafc8e435491dd6ccea04c1034%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7ce2c0e3fa16423ebc7c79cac4a7f9c3%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.655588102462%2C%20-74.01168784060644%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_158d8e5f9e684b6da0b12f02bdaf69fa%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2bb9dbf0e99c4b03af08bacd600e24e7%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2bb9dbf0e99c4b03af08bacd600e24e7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESunset%20Park%20West%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_158d8e5f9e684b6da0b12f02bdaf69fa.setContent%28html_2bb9dbf0e99c4b03af08bacd600e24e7%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7ce2c0e3fa16423ebc7c79cac4a7f9c3.bindPopup%28popup_158d8e5f9e684b6da0b12f02bdaf69fa%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_bae31235070e4283b8103848c4973b3c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.865277077060874%2C%20-73.90762892792475%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a20847a9a1744e3da186a8efcbf40522%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4b74d238a1834cecae572f2ebec276af%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_4b74d238a1834cecae572f2ebec276af%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EKingsbridge%20Heights%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a20847a9a1744e3da186a8efcbf40522.setContent%28html_4b74d238a1834cecae572f2ebec276af%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_bae31235070e4283b8103848c4973b3c.bindPopup%28popup_a20847a9a1744e3da186a8efcbf40522%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_25a7073e83f0474f8bfa0634d1bd93c0%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.85419986286547%2C%20-73.91692050361091%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_82ba3ba52c74493abdd091e02e792971%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_fd2821ca52614ee3aa4b398054dfbe50%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_fd2821ca52614ee3aa4b398054dfbe50%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EUniversity%20Heights-Morris%20Heights%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_82ba3ba52c74493abdd091e02e792971.setContent%28html_fd2821ca52614ee3aa4b398054dfbe50%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_25a7073e83f0474f8bfa0634d1bd93c0.bindPopup%28popup_82ba3ba52c74493abdd091e02e792971%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_869ad3f8a99a43fd9136a62811b39c80%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.63230411248058%2C%20-74.1034612063591%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_18f1761071dd4b8d905f297c4145e899%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f057df8001754189acf23a9b068ce8d9%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_f057df8001754189acf23a9b068ce8d9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENew%20Brighton-Silver%20Lake%2C%20Staten%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_18f1761071dd4b8d905f297c4145e899.setContent%28html_f057df8001754189acf23a9b068ce8d9%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_869ad3f8a99a43fd9136a62811b39c80.bindPopup%28popup_18f1761071dd4b8d905f297c4145e899%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_162748931d7542ccb96900596a90f061%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.640228722852925%2C%20-74.10060870703019%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_48735db6d44648ccafacb43cfd0f7516%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ce5c2d0c291b488e96131e56ca589ac5%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ce5c2d0c291b488e96131e56ca589ac5%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWest%20New%20Brighton-New%20Brighton-St.%20George%2C%20Staten%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_48735db6d44648ccafacb43cfd0f7516.setContent%28html_ce5c2d0c291b488e96131e56ca589ac5%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_162748931d7542ccb96900596a90f061.bindPopup%28popup_48735db6d44648ccafacb43cfd0f7516%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2c73c0ae0c774bce8a3f9fbb94cf410a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.72599095044506%2C%20-73.81869165127154%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e868766ee0f54bc7832946272b1f8c28%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b90c346bcd7e403daeb7568ae58ac62e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b90c346bcd7e403daeb7568ae58ac62e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EKew%20Gardens%20Hills%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e868766ee0f54bc7832946272b1f8c28.setContent%28html_b90c346bcd7e403daeb7568ae58ac62e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2c73c0ae0c774bce8a3f9fbb94cf410a.bindPopup%28popup_e868766ee0f54bc7832946272b1f8c28%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_961eb258deeb49cb99d38789d150a498%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.74084225468238%2C%20-73.85669659382718%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3749cb6e3cb9444e89782224bfda8805%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b530efda21d04e8b82aff517d2d0f0c4%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b530efda21d04e8b82aff517d2d0f0c4%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECorona%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3749cb6e3cb9444e89782224bfda8805.setContent%28html_b530efda21d04e8b82aff517d2d0f0c4%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_961eb258deeb49cb99d38789d150a498.bindPopup%28popup_3749cb6e3cb9444e89782224bfda8805%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_90979d236f5d457493128e5dda9a8e0a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.75436787926768%2C%20-73.86121151628139%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_cb904827302749c9a6d62c91841e2fd8%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_540576bae48e4843a88f3f0a4f618a96%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_540576bae48e4843a88f3f0a4f618a96%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENorth%20Corona%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_cb904827302749c9a6d62c91841e2fd8.setContent%28html_540576bae48e4843a88f3f0a4f618a96%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_90979d236f5d457493128e5dda9a8e0a.bindPopup%28popup_cb904827302749c9a6d62c91841e2fd8%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fb712937047b4c0baea8209eeec9bb8a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.72734350120654%2C%20-73.98635207212405%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_65480a0615424b65b48c05edcc0786a6%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_cba5b81701d9476cb66b9ac925bd160e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_cba5b81701d9476cb66b9ac925bd160e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEast%20Village%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_65480a0615424b65b48c05edcc0786a6.setContent%28html_cba5b81701d9476cb66b9ac925bd160e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_fb712937047b4c0baea8209eeec9bb8a.bindPopup%28popup_65480a0615424b65b48c05edcc0786a6%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c59f468df0d3404fa5b3ee717609020d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.8339519338958%2C%20-73.88366042581148%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4f9d9b7964774e80ab47066468e1f8cb%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6e4d6f22ee454dccb7a0c4ca3bf84ad4%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_6e4d6f22ee454dccb7a0c4ca3bf84ad4%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECrotona%20Park%20East%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4f9d9b7964774e80ab47066468e1f8cb.setContent%28html_6e4d6f22ee454dccb7a0c4ca3bf84ad4%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c59f468df0d3404fa5b3ee717609020d.bindPopup%28popup_4f9d9b7964774e80ab47066468e1f8cb%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ab05014185ae4f38a029fe24aec9dada%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.61621933162528%2C%20-74.12521820953364%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_cd16efa1299f40418ee53fb9dcd19c62%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f1d6ab7e83234fc9a55c2173fa0b315c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_f1d6ab7e83234fc9a55c2173fa0b315c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWesterleigh%2C%20Staten%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_cd16efa1299f40418ee53fb9dcd19c62.setContent%28html_f1d6ab7e83234fc9a55c2173fa0b315c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ab05014185ae4f38a029fe24aec9dada.bindPopup%28popup_cd16efa1299f40418ee53fb9dcd19c62%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3b3349216d62448991e05add5d1564b3%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.641077591494856%2C%20-74.12605854557303%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_730b319574884f33b44c25ff8565402f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_64fc66b7097549d7bf5ed4e1e63dd036%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_64fc66b7097549d7bf5ed4e1e63dd036%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EPort%20Richmond%2C%20Staten%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_730b319574884f33b44c25ff8565402f.setContent%28html_64fc66b7097549d7bf5ed4e1e63dd036%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_3b3349216d62448991e05add5d1564b3.bindPopup%28popup_730b319574884f33b44c25ff8565402f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_bc5c1d2d27bf47a7824a6e003532250e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.705687831163246%2C%20-73.98203911644805%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e7d1b62b50a94fdbab49a909d1181348%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9bb97e6fd7154dc897d9dfe1d6a2c2f5%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9bb97e6fd7154dc897d9dfe1d6a2c2f5%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELower%20East%20Side%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e7d1b62b50a94fdbab49a909d1181348.setContent%28html_9bb97e6fd7154dc897d9dfe1d6a2c2f5%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_bc5c1d2d27bf47a7824a6e003532250e.bindPopup%28popup_e7d1b62b50a94fdbab49a909d1181348%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2f42ab891db24b3ea0f225710067cfe6%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.57980732222715%2C%20-73.96065207075698%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_82034db4ee544ce5805931933f57ac8d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6ef4031447e74c5c95c8711d61ac2c48%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_6ef4031447e74c5c95c8711d61ac2c48%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBrighton%20Beach%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_82034db4ee544ce5805931933f57ac8d.setContent%28html_6ef4031447e74c5c95c8711d61ac2c48%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2f42ab891db24b3ea0f225710067cfe6.bindPopup%28popup_82034db4ee544ce5805931933f57ac8d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3e6156c5a7c44e2d893ae4b42bb86519%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.57962642534297%2C%20-73.97222298021403%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3ba7bf4e1ef948f49957bb1f9ae869c9%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6dde5eebcd0b4e8c8d5e8b2521dcc3bf%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_6dde5eebcd0b4e8c8d5e8b2521dcc3bf%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWest%20Brighton%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3ba7bf4e1ef948f49957bb1f9ae869c9.setContent%28html_6dde5eebcd0b4e8c8d5e8b2521dcc3bf%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_3e6156c5a7c44e2d893ae4b42bb86519.bindPopup%28popup_3ba7bf4e1ef948f49957bb1f9ae869c9%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_44e1baa0345641509887efdd01bef105%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.72077408216818%2C%20-73.84446125410432%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_cf9abf7b77714116a0972f09b141d9e0%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9455c9e9f752452685b60da13e664ca0%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9455c9e9f752452685b60da13e664ca0%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EForest%20Hills%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_cf9abf7b77714116a0972f09b141d9e0.setContent%28html_9455c9e9f752452685b60da13e664ca0%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_44e1baa0345641509887efdd01bef105.bindPopup%28popup_cf9abf7b77714116a0972f09b141d9e0%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4e1fe86d79bc437c8ab8dc92c38e060b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.726102470785946%2C%20-73.86400846832662%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_84c164ea7fa541528274a551dfc1543a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_409dd59010bb4e82998ce7f646d99eca%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_409dd59010bb4e82998ce7f646d99eca%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERego%20Park%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_84c164ea7fa541528274a551dfc1543a.setContent%28html_409dd59010bb4e82998ce7f646d99eca%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4e1fe86d79bc437c8ab8dc92c38e060b.bindPopup%28popup_84c164ea7fa541528274a551dfc1543a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_db3ac82c52334241a5d88210d1805874%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.73982212741276%2C%20-73.97474305311528%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_dc12c1f521e44122bf144d014f22182f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a8a147ff64aa4a7784d95526ee459c10%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_a8a147ff64aa4a7784d95526ee459c10%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMurray%20Hill-Kips%20Bay%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_dc12c1f521e44122bf144d014f22182f.setContent%28html_a8a147ff64aa4a7784d95526ee459c10%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_db3ac82c52334241a5d88210d1805874.bindPopup%28popup_dc12c1f521e44122bf144d014f22182f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4a8e70623302492281bef114507fbaa9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.759158838430096%2C%20-73.88737200916617%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_93de73b93fde4d58b6a23349fc95111c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5416bf1cd4784cc1944d272ebe4a9e0c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_5416bf1cd4784cc1944d272ebe4a9e0c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EJackson%20Heights%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_93de73b93fde4d58b6a23349fc95111c.setContent%28html_5416bf1cd4784cc1944d272ebe4a9e0c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4a8e70623302492281bef114507fbaa9.bindPopup%28popup_93de73b93fde4d58b6a23349fc95111c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_88f325e8f0404b6ba7a17c3328cb0f0a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.75048120799816%2C%20-73.90405155375089%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_0491336ca1124ff991c5a0cb4e976000%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8933ad78771b4158ac34d37f9de32897%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8933ad78771b4158ac34d37f9de32897%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWoodside%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_0491336ca1124ff991c5a0cb4e976000.setContent%28html_8933ad78771b4158ac34d37f9de32897%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_88f325e8f0404b6ba7a17c3328cb0f0a.bindPopup%28popup_0491336ca1124ff991c5a0cb4e976000%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f4e71c0ec1e948aca6e100de7e746777%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.831998741148844%2C%20-73.9139440031317%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ebd6bb6b3dbe42c59faa3bf02cfecdce%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9c5f3a272936403ba0adbc28fd8547f3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9c5f3a272936403ba0adbc28fd8547f3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEast%20Concourse-Concourse%20Village%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ebd6bb6b3dbe42c59faa3bf02cfecdce.setContent%28html_9c5f3a272936403ba0adbc28fd8547f3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f4e71c0ec1e948aca6e100de7e746777.bindPopup%28popup_ebd6bb6b3dbe42c59faa3bf02cfecdce%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ec33999147284d72a7c102bae8cd5286%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.7122678747341%2C%20-73.80744599764525%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4915e2b5351c401fb3f347b2968c96c0%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d1c4333ba91b4312a564c17d7d1c1d38%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d1c4333ba91b4312a564c17d7d1c1d38%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBriarwood-Jamaica%20Hills%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4915e2b5351c401fb3f347b2968c96c0.setContent%28html_d1c4333ba91b4312a564c17d7d1c1d38%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ec33999147284d72a7c102bae8cd5286.bindPopup%28popup_4915e2b5351c401fb3f347b2968c96c0%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0f1046bc489d44bb9cfb26d46d7eb7dd%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.70853666686447%2C%20-73.83035481228406%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1a3191dcb6a0449f80ee67b802e7d33f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_dcdb48218191425eb952896b528429f8%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_dcdb48218191425eb952896b528429f8%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EKew%20Gardens%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1a3191dcb6a0449f80ee67b802e7d33f.setContent%28html_dcdb48218191425eb952896b528429f8%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_0f1046bc489d44bb9cfb26d46d7eb7dd.bindPopup%28popup_1a3191dcb6a0449f80ee67b802e7d33f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_aee5919994874130aa991faf140154df%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.7615482696381%2C%20-73.91878760289262%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_8c829288e524408aa45bca72108093a7%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a423212decb34c8eab71211fd8599cfa%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_a423212decb34c8eab71211fd8599cfa%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EAstoria%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_8c829288e524408aa45bca72108093a7.setContent%28html_a423212decb34c8eab71211fd8599cfa%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_aee5919994874130aa991faf140154df.bindPopup%28popup_8c829288e524408aa45bca72108093a7%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c74381f9756347f2949d99afa1671fdd%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.74308896487988%2C%20-73.8198020849463%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b6a92f129c574ac1aa2745009a48a801%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5dae3946229e4dba941fd0aa27b740c0%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_5dae3946229e4dba941fd0aa27b740c0%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EQueensboro%20Hill%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b6a92f129c574ac1aa2745009a48a801.setContent%28html_5dae3946229e4dba941fd0aa27b740c0%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c74381f9756347f2949d99afa1671fdd.bindPopup%28popup_b6a92f129c574ac1aa2745009a48a801%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b40011245c2e47e687f52b435a87d0dc%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.73459251181328%2C%20-73.9490716513249%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_add1e4c3409448aa864ee627b35f314b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e84acad72fdb437bac13eb0596d8d864%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e84acad72fdb437bac13eb0596d8d864%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGreenpoint%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_add1e4c3409448aa864ee627b35f314b.setContent%28html_e84acad72fdb437bac13eb0596d8d864%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b40011245c2e47e687f52b435a87d0dc.bindPopup%28popup_add1e4c3409448aa864ee627b35f314b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c5455938cddd402487b1f1c2f4230d5b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.88239049251526%2C%20-73.85922810026348%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_bea2420f6ebe49e5be12bf44fd759f12%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_29e876f6298f4ccdb6c7346d5bde626b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_29e876f6298f4ccdb6c7346d5bde626b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWilliamsbridge-Olinville%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_bea2420f6ebe49e5be12bf44fd759f12.setContent%28html_29e876f6298f4ccdb6c7346d5bde626b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c5455938cddd402487b1f1c2f4230d5b.bindPopup%28popup_bea2420f6ebe49e5be12bf44fd759f12%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_85e6d33af4404a1fa713fc6c59372e1a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.64260518382003%2C%20-74.15940098324363%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4f65a689426e430c8162c2babcd2460b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c61b88d41f664f96a015cc246a5cc0ba%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c61b88d41f664f96a015cc246a5cc0ba%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMariner%26%2339%3Bs%20Harbor-Arlington-Port%20Ivory-Graniteville%2C%20Staten%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4f65a689426e430c8162c2babcd2460b.setContent%28html_c61b88d41f664f96a015cc246a5cc0ba%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_85e6d33af4404a1fa713fc6c59372e1a.bindPopup%28popup_4f65a689426e430c8162c2babcd2460b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_132e569e9ae74cc896fc27f8d5423f19%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.808454248987175%2C%20-73.96406710398044%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_00c2226b6b6647309afc1038f39d7930%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6cfb170088574b878557f1eb019ab546%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_6cfb170088574b878557f1eb019ab546%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMorningside%20Heights%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_00c2226b6b6647309afc1038f39d7930.setContent%28html_6cfb170088574b878557f1eb019ab546%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_132e569e9ae74cc896fc27f8d5423f19.bindPopup%28popup_00c2226b6b6647309afc1038f39d7930%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3fcef05529174c41863d1ec2b9c145d8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.59691442502093%2C%20-73.79123141465122%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_9fae92212b37459db942f7312caf85a0%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6ba9cbcbcd3040a4b4d8f9909763a2f3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_6ba9cbcbcd3040a4b4d8f9909763a2f3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHammels-Arverne-Edgemere%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_9fae92212b37459db942f7312caf85a0.setContent%28html_6ba9cbcbcd3040a4b4d8f9909763a2f3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_3fcef05529174c41863d1ec2b9c145d8.bindPopup%28popup_9fae92212b37459db942f7312caf85a0%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_da6d08abcdaa4477aeae38a5ab3ff287%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.60278238343513%2C%20-73.75758084269403%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_461dc253e2374575bcd0f6ff7d52d295%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e2de09d5d2294aeea4bf02445d92a0f5%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e2de09d5d2294aeea4bf02445d92a0f5%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFar%20Rockaway-Bayswater%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_461dc253e2374575bcd0f6ff7d52d295.setContent%28html_e2de09d5d2294aeea4bf02445d92a0f5%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_da6d08abcdaa4477aeae38a5ab3ff287.bindPopup%28popup_461dc253e2374575bcd0f6ff7d52d295%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c404eedb92bd4f53a38f752a5928e9ce%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.84323111220925%2C%20-73.90086445676883%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_9e0a036855384c3a8cbb888c2e1bbad7%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_02c0946a1f6e4f98a6f3d5db0b3d62ec%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_02c0946a1f6e4f98a6f3d5db0b3d62ec%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EClaremont-Bathgate%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_9e0a036855384c3a8cbb888c2e1bbad7.setContent%28html_02c0946a1f6e4f98a6f3d5db0b3d62ec%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c404eedb92bd4f53a38f752a5928e9ce.bindPopup%28popup_9e0a036855384c3a8cbb888c2e1bbad7%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e93d1e0b066b4152b79cf4c18f61f533%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.85765544136461%2C%20-73.93734644214263%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_144c1e85fb5643008d02ad174bf88cdd%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c3b62d6ac5834648b205308bf6ba5f1e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c3b62d6ac5834648b205308bf6ba5f1e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWashington%20Heights%20North%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_144c1e85fb5643008d02ad174bf88cdd.setContent%28html_c3b62d6ac5834648b205308bf6ba5f1e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e93d1e0b066b4152b79cf4c18f61f533.bindPopup%28popup_144c1e85fb5643008d02ad174bf88cdd%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c1e0ba3fff784d37a771e4464b5e67c2%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.84274234313284%2C%20-73.9433111501022%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b5ff0a1d26ae4c9f81b40cda75f65102%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_29a32d8c7eab4781acc10f683169c7e3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_29a32d8c7eab4781acc10f683169c7e3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWashington%20Heights%20South%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b5ff0a1d26ae4c9f81b40cda75f65102.setContent%28html_29a32d8c7eab4781acc10f683169c7e3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c1e0ba3fff784d37a771e4464b5e67c2.bindPopup%28popup_b5ff0a1d26ae4c9f81b40cda75f65102%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2f43bf7d909a4695ad970cb845a72272%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.84846028981872%2C%20-73.8491012816575%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_73bebcade6424484b045478ac1ae4e31%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_29c5b7fc68db4848b743052534ca59cf%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_29c5b7fc68db4848b743052534ca59cf%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EVan%20Nest-Morris%20Park-Westchester%20Square%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_73bebcade6424484b045478ac1ae4e31.setContent%28html_29c5b7fc68db4848b743052534ca59cf%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2f43bf7d909a4695ad970cb845a72272.bindPopup%28popup_73bebcade6424484b045478ac1ae4e31%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_52d100d2eaa544149bc3ee66de628d3f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.83436038778096%2C%20-73.84493704337497%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_9efb361eac134695a680f6fbcb0b8600%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_201420d06bf6437ea08a76b3c322fe21%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_201420d06bf6437ea08a76b3c322fe21%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWestchester-Unionport%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_9efb361eac134695a680f6fbcb0b8600.setContent%28html_201420d06bf6437ea08a76b3c322fe21%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_52d100d2eaa544149bc3ee66de628d3f.bindPopup%28popup_9efb361eac134695a680f6fbcb0b8600%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0a5e4737fff34ef391164c00ef746bc4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.81975438504283%2C%20-73.91352804016817%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_370eba3cedc949da985a474cec0f9b30%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_df254ce3a3844e47b0a51d47b4b70b9f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_df254ce3a3844e47b0a51d47b4b70b9f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMelrose%20South-Mott%20Haven%20North%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_370eba3cedc949da985a474cec0f9b30.setContent%28html_df254ce3a3844e47b0a51d47b4b70b9f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_0a5e4737fff34ef391164c00ef746bc4.bindPopup%28popup_370eba3cedc949da985a474cec0f9b30%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f3fcf56adf154c6e817d071e3d2dbd30%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.621623932582864%2C%20-74.01447425224704%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f3518a93b36041fab19c306dba7d6324%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a5507d37af7b4cfbaa436a84023e3b5d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_a5507d37af7b4cfbaa436a84023e3b5d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDyker%20Heights%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f3518a93b36041fab19c306dba7d6324.setContent%28html_a5507d37af7b4cfbaa436a84023e3b5d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f3fcf56adf154c6e817d071e3d2dbd30.bindPopup%28popup_f3518a93b36041fab19c306dba7d6324%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b2cbb7c65a374d13b273951efc9cf8b8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.62535721666878%2C%20-74.02877360158438%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_385564d740a94b248aa8cff57125bb0d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_dfa99d8d0cef401286faf0e8d703a22c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_dfa99d8d0cef401286faf0e8d703a22c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBay%20Ridge%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_385564d740a94b248aa8cff57125bb0d.setContent%28html_dfa99d8d0cef401286faf0e8d703a22c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b2cbb7c65a374d13b273951efc9cf8b8.bindPopup%28popup_385564d740a94b248aa8cff57125bb0d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7a052c1f7f554f7bb03d5102e70ed4a4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.58898060217025%2C%20-74.1958382188466%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_75ffd1c069a14c2b9dbbb41caa2e6e72%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d867449ef5604884bbb0d7fea319d9fd%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d867449ef5604884bbb0d7fea319d9fd%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENew%20Springville-Bloomfield-Travis%2C%20Staten%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_75ffd1c069a14c2b9dbbb41caa2e6e72.setContent%28html_d867449ef5604884bbb0d7fea319d9fd%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7a052c1f7f554f7bb03d5102e70ed4a4.bindPopup%28popup_75ffd1c069a14c2b9dbbb41caa2e6e72%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a213b27d6aa54e89a2d5202974eccee8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.564536526440385%2C%20-74.12191848055942%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_7c7e0818194b4489a4e1661eae4582f8%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6ef38ea49a5947cb89911f9bd3205f4d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_6ef38ea49a5947cb89911f9bd3205f4d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EOakwood-Oakwood%20Beach%2C%20Staten%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_7c7e0818194b4489a4e1661eae4582f8.setContent%28html_6ef38ea49a5947cb89911f9bd3205f4d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a213b27d6aa54e89a2d5202974eccee8.bindPopup%28popup_7c7e0818194b4489a4e1661eae4582f8%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c225a42624974cd1aa9388816b846a6d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.54314093074463%2C%20-74.14359092498653%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d75909a4918245a380584f3925b598ad%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6f8db45c2b6d4e3d90ae3c6f03f32755%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_6f8db45c2b6d4e3d90ae3c6f03f32755%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGreat%20Kills%2C%20Staten%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d75909a4918245a380584f3925b598ad.setContent%28html_6f8db45c2b6d4e3d90ae3c6f03f32755%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c225a42624974cd1aa9388816b846a6d.bindPopup%28popup_d75909a4918245a380584f3925b598ad%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a7fb95ea1aa8470386a3e607b58b750b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.60097354925467%2C%20-74.07261144262483%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_2149f8e63e2b4feb8ccdf6e384b3cfe9%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_46ed9b1712fe4b2f90e6992c3cc60477%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_46ed9b1712fe4b2f90e6992c3cc60477%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGrasmere-Arrochar-Ft.%20Wadsworth%2C%20Staten%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_2149f8e63e2b4feb8ccdf6e384b3cfe9.setContent%28html_46ed9b1712fe4b2f90e6992c3cc60477%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a7fb95ea1aa8470386a3e607b58b750b.bindPopup%28popup_2149f8e63e2b4feb8ccdf6e384b3cfe9%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7d35ef898eee4dd498a3d25d421d2b00%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.620675047563395%2C%20-74.07246079160275%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_722c7fb7085449269c56aa9fd67988e8%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_cb7fcb2c50ba4146a151c9fb3b7bf1a8%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_cb7fcb2c50ba4146a151c9fb3b7bf1a8%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EStapleton-Rosebank%2C%20Staten%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_722c7fb7085449269c56aa9fd67988e8.setContent%28html_cb7fcb2c50ba4146a151c9fb3b7bf1a8%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7d35ef898eee4dd498a3d25d421d2b00.bindPopup%28popup_722c7fb7085449269c56aa9fd67988e8%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1a059eb631fb4a7ba2385f07d3cd40c0%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.73426270771225%2C%20-73.7284248039466%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6b254a34812d4c74b5fca3d872da34fc%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a2648769baf54a64903bbfc8b5925804%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_a2648769baf54a64903bbfc8b5925804%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBellerose%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6b254a34812d4c74b5fca3d872da34fc.setContent%28html_a2648769baf54a64903bbfc8b5925804%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_1a059eb631fb4a7ba2385f07d3cd40c0.bindPopup%28popup_6b254a34812d4c74b5fca3d872da34fc%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1db1d4805aa14166a763b13c87b902a5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.746334256862774%2C%20-73.71568391912807%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_614cebcb0c70429988010c116748b729%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1c8c90b72faf4c7593be0b930a50ea7f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1c8c90b72faf4c7593be0b930a50ea7f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGlen%20Oaks-Floral%20Park-New%20Hyde%20Park%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_614cebcb0c70429988010c116748b729.setContent%28html_1c8c90b72faf4c7593be0b930a50ea7f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_1db1d4805aa14166a763b13c87b902a5.bindPopup%28popup_614cebcb0c70429988010c116748b729%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_806b97142f7c469497808e07b98e8e6c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.69459452313694%2C%20-73.79216165246254%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_0ad05700ef54425eab5e4c5e5d4b8d22%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_dfee678a28464272b96ffb44cdb484ce%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_dfee678a28464272b96ffb44cdb484ce%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESouth%20Jamaica%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_0ad05700ef54425eab5e4c5e5d4b8d22.setContent%28html_dfee678a28464272b96ffb44cdb484ce%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_806b97142f7c469497808e07b98e8e6c.bindPopup%28popup_0ad05700ef54425eab5e4c5e5d4b8d22%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b04f2b932b5040c295455d63ff330290%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.74880110296486%2C%20-73.96949297606015%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_aa0a1946d03d4cdab3d9756c84190bce%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f67bef03ec7344489f87d2638714c093%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_f67bef03ec7344489f87d2638714c093%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ETurtle%20Bay-East%20Midtown%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_aa0a1946d03d4cdab3d9756c84190bce.setContent%28html_f67bef03ec7344489f87d2638714c093%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b04f2b932b5040c295455d63ff330290.bindPopup%28popup_aa0a1946d03d4cdab3d9756c84190bce%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3111756c50a546659ca0696400d6aeb7%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.748395140173336%2C%20-73.95764383283307%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_749b547221264a58bc5820df8dc9ff73%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ee50cc5fd07f49b8af2e1e11300f389d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ee50cc5fd07f49b8af2e1e11300f389d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELenox%20Hill-Roosevelt%20Island%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_749b547221264a58bc5820df8dc9ff73.setContent%28html_ee50cc5fd07f49b8af2e1e11300f389d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_3111756c50a546659ca0696400d6aeb7.bindPopup%28popup_749b547221264a58bc5820df8dc9ff73%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7b3f2b1825e94a28a0978381f43d7354%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.73212325034415%2C%20-73.93565968707185%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5b1b0ec310e146a59cea53bfb05c271c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d1c957f20c7046f0a9226080c814e379%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d1c957f20c7046f0a9226080c814e379%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHunters%20Point-Sunnyside-West%20Maspeth%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5b1b0ec310e146a59cea53bfb05c271c.setContent%28html_d1c957f20c7046f0a9226080c814e379%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7b3f2b1825e94a28a0978381f43d7354.bindPopup%28popup_5b1b0ec310e146a59cea53bfb05c271c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_aab993c382894222a5b1a109763dea3c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.63715128580007%2C%20-73.74708238727378%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_bc32ccd7be0848af812d3e956f9f5bf1%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d0442b96b6c6446fbc6f759fffc32003%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d0442b96b6c6446fbc6f759fffc32003%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EAirport%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_bc32ccd7be0848af812d3e956f9f5bf1.setContent%28html_d0442b96b6c6446fbc6f759fffc32003%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_aab993c382894222a5b1a109763dea3c.bindPopup%28popup_bc32ccd7be0848af812d3e956f9f5bf1%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_99480b4680e44e1f89047efaa2a44d39%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.68779374760685%2C%20-73.85481035253551%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ef7305cc01bb41c09450752f689ad14f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_281fbc8c83c547bbb0be6010aae2c934%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_281fbc8c83c547bbb0be6010aae2c934%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWoodhaven%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ef7305cc01bb41c09450752f689ad14f.setContent%28html_281fbc8c83c547bbb0be6010aae2c934%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_99480b4680e44e1f89047efaa2a44d39.bindPopup%28popup_ef7305cc01bb41c09450752f689ad14f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fa43827c68e345a2a6ce2d493d9018b8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.79497803542242%2C%20-73.81536969015437%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5da968e27a4f432b992ea7d400915739%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_50461615b7db4d228c9f973c898d9582%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_50461615b7db4d228c9f973c898d9582%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWhitestone%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5da968e27a4f432b992ea7d400915739.setContent%28html_50461615b7db4d228c9f973c898d9582%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_fa43827c68e345a2a6ce2d493d9018b8.bindPopup%28popup_5da968e27a4f432b992ea7d400915739%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_01db2eed8fd245d5b9eec79b47b468b3%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.674978738476575%2C%20-73.93584743631712%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_09929b4116bf42febd5f319d3575f24e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e29fafa6ba51407d96eb38012d22f413%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e29fafa6ba51407d96eb38012d22f413%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECrown%20Heights%20North%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_09929b4116bf42febd5f319d3575f24e.setContent%28html_e29fafa6ba51407d96eb38012d22f413%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_01db2eed8fd245d5b9eec79b47b468b3.bindPopup%28popup_09929b4116bf42febd5f319d3575f24e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_8237f5aaaad34ae5a2516235503c7faf%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.676047521170005%2C%20-73.96762654167343%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d8f6329ef06247bb9b79fd28f8dd3cf9%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f936f0eba63542fda236ec599e377fa8%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_f936f0eba63542fda236ec599e377fa8%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EProspect%20Heights%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d8f6329ef06247bb9b79fd28f8dd3cf9.setContent%28html_f936f0eba63542fda236ec599e377fa8%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_8237f5aaaad34ae5a2516235503c7faf.bindPopup%28popup_d8f6329ef06247bb9b79fd28f8dd3cf9%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_882c801bc79348899fb08e0e51ac9d03%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.67616549653592%2C%20-73.84838810961706%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_99839017341a4e5caf5a25b4d3a58fca%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b7d23179eaf54204ab8eeaab918e8bf8%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b7d23179eaf54204ab8eeaab918e8bf8%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EOzone%20Park%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_99839017341a4e5caf5a25b4d3a58fca.setContent%28html_b7d23179eaf54204ab8eeaab918e8bf8%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_882c801bc79348899fb08e0e51ac9d03.bindPopup%28popup_99839017341a4e5caf5a25b4d3a58fca%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_8a60af04f1e74370a6d3e8b515c59798%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.65252553774554%2C%20-73.85868482960555%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e320df3f774a46b1845b5314a1f5dc7c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_660d9879fc374cb4bf6d4c77b267c78a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_660d9879fc374cb4bf6d4c77b267c78a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELindenwood-Howard%20Beach%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e320df3f774a46b1845b5314a1f5dc7c.setContent%28html_660d9879fc374cb4bf6d4c77b267c78a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_8a60af04f1e74370a6d3e8b515c59798.bindPopup%28popup_e320df3f774a46b1845b5314a1f5dc7c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fe580bb290b54a11ade770716b33a9bf%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.58387926349231%2C%20-73.91762470237235%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_082724ff1af04686b713826891da47a3%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_98d81a339e114069a6f91e5a109d654b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_98d81a339e114069a6f91e5a109d654b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3Epark-cemetery-etc-Brooklyn%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_082724ff1af04686b713826891da47a3.setContent%28html_98d81a339e114069a6f91e5a109d654b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_fe580bb290b54a11ade770716b33a9bf.bindPopup%28popup_082724ff1af04686b713826891da47a3%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4157e61c48414906ad069ac1cb1599d2%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.695942986842866%2C%20-73.82947802747712%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_06605fbd7f6142cb9baf60def5018651%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8dd06a3db36f4e6c99f33eecf3095fbe%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8dd06a3db36f4e6c99f33eecf3095fbe%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERichmond%20Hill%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_06605fbd7f6142cb9baf60def5018651.setContent%28html_8dd06a3db36f4e6c99f33eecf3095fbe%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4157e61c48414906ad069ac1cb1599d2.bindPopup%28popup_06605fbd7f6142cb9baf60def5018651%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3f200fa0d3fb4da0a57773a07e5d9c86%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.70616601170704%2C%20-73.79052166578737%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ae5fa0c176c74e3997ca1c3538659574%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_17061fd441c6484eadb03c1d2b0ba7fe%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_17061fd441c6484eadb03c1d2b0ba7fe%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EJamaica%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ae5fa0c176c74e3997ca1c3538659574.setContent%28html_17061fd441c6484eadb03c1d2b0ba7fe%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_3f200fa0d3fb4da0a57773a07e5d9c86.bindPopup%28popup_ae5fa0c176c74e3997ca1c3538659574%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d8dc61458bb9490a8ef6166886b45943%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.562108607086216%2C%20-73.90755443120086%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1d193265929d48a39eebc2ba2b2fae66%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8a26191b163a4f7283e864a7f845cbaa%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8a26191b163a4f7283e864a7f845cbaa%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBreezy%20Point-Belle%20Harbor-Rockaway%20Park-Broad%20Channel%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1d193265929d48a39eebc2ba2b2fae66.setContent%28html_8a26191b163a4f7283e864a7f845cbaa%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_d8dc61458bb9490a8ef6166886b45943.bindPopup%28popup_1d193265929d48a39eebc2ba2b2fae66%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c3e275470af54ed98802c2ad75d4fe37%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.56141613992784%2C%20-73.88559493145073%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d2469927cfbd4afeaaa74540371dd3d2%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8f22ff1e2cab4de484dbacaf19fc42bc%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8f22ff1e2cab4de484dbacaf19fc42bc%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3Epark-cemetery-etc-Queens%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d2469927cfbd4afeaaa74540371dd3d2.setContent%28html_8f22ff1e2cab4de484dbacaf19fc42bc%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c3e275470af54ed98802c2ad75d4fe37.bindPopup%28popup_d2469927cfbd4afeaaa74540371dd3d2%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_807bf13eb9164d7e9880c9c2f3f723eb%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.834902908756284%2C%20-73.78879067260218%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c40ed062e57940f68c152ce09e05fac7%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4ca9ee8cf547471badee1611ec1a44c2%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_4ca9ee8cf547471badee1611ec1a44c2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EPelham%20Bay-Country%20Club-City%20Island%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c40ed062e57940f68c152ce09e05fac7.setContent%28html_4ca9ee8cf547471badee1611ec1a44c2%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_807bf13eb9164d7e9880c9c2f3f723eb.bindPopup%28popup_c40ed062e57940f68c152ce09e05fac7%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9dd181eaf60a4fba9165dbd3a14ff07e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.81916677276425%2C%20-73.87092261946509%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b7ca3a08fdca4a589635201a215fd9c2%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_dd9c80de58e64339a1f263cc14613e9d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_dd9c80de58e64339a1f263cc14613e9d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3Epark-cemetery-etc-Bronx%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b7ca3a08fdca4a589635201a215fd9c2.setContent%28html_dd9c80de58e64339a1f263cc14613e9d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9dd181eaf60a4fba9165dbd3a14ff07e.bindPopup%28popup_b7ca3a08fdca4a589635201a215fd9c2%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_8eb126036ddc44a49fafad3e0ba0f9ce%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.80776467946547%2C%20-73.88937480322028%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_7a6ffe31c34f4d57ade60e4e4f30b5db%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_69f6d629017e4e139516cdec3faf17dd%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_69f6d629017e4e139516cdec3faf17dd%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHunts%20Point%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_7a6ffe31c34f4d57ade60e4e4f30b5db.setContent%28html_69f6d629017e4e139516cdec3faf17dd%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_8eb126036ddc44a49fafad3e0ba0f9ce.bindPopup%28popup_7a6ffe31c34f4d57ade60e4e4f30b5db%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5742d5c2e0f147b0a8693da09e451b5a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.68115225367788%2C%20-73.88233353850096%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d6fb4a4248424132b031d679b8ae87b4%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1d7093c8a09348a7b9932f25c6c77824%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1d7093c8a09348a7b9932f25c6c77824%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECypress%20Hills-City%20Line%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d6fb4a4248424132b031d679b8ae87b4.setContent%28html_1d7093c8a09348a7b9932f25c6c77824%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_5742d5c2e0f147b0a8693da09e451b5a.bindPopup%28popup_d6fb4a4248424132b031d679b8ae87b4%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_944bc3855d434182b3205dc33beb57b5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.804946986223875%2C%20-73.91325242582134%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_158dbaa2c5bd4dfdaebf6a54919838ed%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b4d3d0ff34864f36b3e79501fa830496%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b4d3d0ff34864f36b3e79501fa830496%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMott%20Haven-Port%20Morris%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_158dbaa2c5bd4dfdaebf6a54919838ed.setContent%28html_b4d3d0ff34864f36b3e79501fa830496%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_944bc3855d434182b3205dc33beb57b5.bindPopup%28popup_158dbaa2c5bd4dfdaebf6a54919838ed%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_22a9a4dbde77444c9673d175666f6566%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.830910353286534%2C%20-73.92368688207982%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_2d6d29da72474dca987eecf6bffa0823%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2c4a16953207460787298334a08ede3f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2c4a16953207460787298334a08ede3f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWest%20Concourse%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_2d6d29da72474dca987eecf6bffa0823.setContent%28html_2c4a16953207460787298334a08ede3f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_22a9a4dbde77444c9673d175666f6566.bindPopup%28popup_2d6d29da72474dca987eecf6bffa0823%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_8e522ec3a5874ea1a3aa15db0fda1d99%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.7372057386339%2C%20-73.87941291862147%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_0f1ab46cfafc40149cb7db99b0c29327%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c7b61f53f2e2460b98219e70907df603%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c7b61f53f2e2460b98219e70907df603%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EElmhurst%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_0f1ab46cfafc40149cb7db99b0c29327.setContent%28html_c7b61f53f2e2460b98219e70907df603%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_8e522ec3a5874ea1a3aa15db0fda1d99.bindPopup%28popup_0f1ab46cfafc40149cb7db99b0c29327%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_65d98417a2c44477b235c9367558b52c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.73824917338046%2C%20-73.89284569557071%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4858f80010af497191bdc5de35863c3a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_82376b765e0249ccafcece9c520349f9%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_82376b765e0249ccafcece9c520349f9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EElmhurst-Maspeth%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4858f80010af497191bdc5de35863c3a.setContent%28html_82376b765e0249ccafcece9c520349f9%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_65d98417a2c44477b235c9367558b52c.bindPopup%28popup_4858f80010af497191bdc5de35863c3a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_03add0f37b00408fadd7ece7a9487599%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.58873404099965%2C%20-74.12829097915801%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5957226354fe4d68bf11e5f74737c563%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_af718ff0cc1c40a4a1ba13aaa30781a8%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_af718ff0cc1c40a4a1ba13aaa30781a8%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ETodt%20Hill-Emerson%20Hill-Heartland%20Village-Lighthouse%20Hill%2C%20Staten%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5957226354fe4d68bf11e5f74737c563.setContent%28html_af718ff0cc1c40a4a1ba13aaa30781a8%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_03add0f37b00408fadd7ece7a9487599.bindPopup%28popup_5957226354fe4d68bf11e5f74737c563%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3b6ec98eba384df0a918b73c91274d0e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.58420519729345%2C%20-74.0796019114537%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_58375dec8c644087b837a8b58a499941%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9d1c504ee5404a5ba988cb84c489d796%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9d1c504ee5404a5ba988cb84c489d796%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EOld%20Town-Dongan%20Hills-South%20Beach%2C%20Staten%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_58375dec8c644087b837a8b58a499941.setContent%28html_9d1c504ee5404a5ba988cb84c489d796%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_3b6ec98eba384df0a918b73c91274d0e.bindPopup%28popup_58375dec8c644087b837a8b58a499941%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d87090f4e4264c0f8389dc137dffe458%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.54045501594299%2C%20-74.12993557332341%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_66159ed921e2470ca79d72ebcac4e32f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2fffbf2c027349bfac6b3c4466f8d7ce%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2fffbf2c027349bfac6b3c4466f8d7ce%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3Epark-cemetery-etc-Staten%20Island%2C%20Staten%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_66159ed921e2470ca79d72ebcac4e32f.setContent%28html_2fffbf2c027349bfac6b3c4466f8d7ce%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_d87090f4e4264c0f8389dc137dffe458.bindPopup%28popup_66159ed921e2470ca79d72ebcac4e32f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f374fb91c51842528787207060e25be5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.66218169883386%2C%20-73.77064021566247%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_72541293be2b4de4bc407fcbfc8efd5c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_83603138b2c043edba0d52b3ca1c87bf%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_83603138b2c043edba0d52b3ca1c87bf%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESpringfield%20Gardens%20South-Brookville%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_72541293be2b4de4bc407fcbfc8efd5c.setContent%28html_83603138b2c043edba0d52b3ca1c87bf%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f374fb91c51842528787207060e25be5.bindPopup%28popup_72541293be2b4de4bc407fcbfc8efd5c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_746e645c48584d428ecc96639ac1893d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.67768390718916%2C%20-73.79034727131615%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6569295802044f15ae2a54ead0f9f10e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_41093589d3ff4b5bb42c6a41b3735fc4%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_41093589d3ff4b5bb42c6a41b3735fc4%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBaisley%20Park%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6569295802044f15ae2a54ead0f9f10e.setContent%28html_41093589d3ff4b5bb42c6a41b3735fc4%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_746e645c48584d428ecc96639ac1893d.bindPopup%28popup_6569295802044f15ae2a54ead0f9f10e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_020401afef65460e95cd23cdb1d2b5c0%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.67296041620256%2C%20-73.77318304915985%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_8de816732aae4f5899561e2ca418dd41%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3be6183275924e039fa1513f0d0ccae5%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_3be6183275924e039fa1513f0d0ccae5%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESpringfield%20Gardens%20North%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_8de816732aae4f5899561e2ca418dd41.setContent%28html_3be6183275924e039fa1513f0d0ccae5%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_020401afef65460e95cd23cdb1d2b5c0.bindPopup%28popup_8de816732aae4f5899561e2ca418dd41%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e852b5cbc2bb4a6fa5da81cb019328b9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.67656976894305%2C%20-73.74435128560636%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3ffab062189442848ab953ae40fd5526%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7a6b5a930c0b4a469169a4c82826189e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7a6b5a930c0b4a469169a4c82826189e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELaurelton%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3ffab062189442848ab953ae40fd5526.setContent%28html_7a6b5a930c0b4a469169a4c82826189e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e852b5cbc2bb4a6fa5da81cb019328b9.bindPopup%28popup_3ffab062189442848ab953ae40fd5526%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4d96ddd921a14986a9d233bbcd48fadf%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.71899728094796%2C%20-73.77871544605621%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_175702ab29b04b1caeb074eceaae7eae%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c0865cb2656444fa948ea890508c6015%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c0865cb2656444fa948ea890508c6015%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EJamaica%20Estates-Holliswood%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_175702ab29b04b1caeb074eceaae7eae.setContent%28html_c0865cb2656444fa948ea890508c6015%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4d96ddd921a14986a9d233bbcd48fadf.bindPopup%28popup_175702ab29b04b1caeb074eceaae7eae%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7980f59249994cffb61e35a4a37d9245%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.72833882688117%2C%20-73.80542255271689%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d35c36fb52b94650983fd24386925c95%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5602c0159d15441896a668014d9a964d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_5602c0159d15441896a668014d9a964d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EPomonok-Flushing%20Heights-Hillcrest%2C%20Queens%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d35c36fb52b94650983fd24386925c95.setContent%28html_5602c0159d15441896a668014d9a964d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7980f59249994cffb61e35a4a37d9245.bindPopup%28popup_d35c36fb52b94650983fd24386925c95%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_8c46533253304165a4e6e984c2b76364%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.72044868765068%2C%20-74.0071853926249%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_8802477f00944675b0fc5c11bc7ad023%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_11863d8c275946878715ed53a67a5051%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_11863d8c275946878715ed53a67a5051%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESoHo-TriBeCa-Civic%20Center-Little%20Italy%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_8802477f00944675b0fc5c11bc7ad023.setContent%28html_11863d8c275946878715ed53a67a5051%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_8c46533253304165a4e6e984c2b76364.bindPopup%28popup_8802477f00944675b0fc5c11bc7ad023%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_579e08eb8b964f8789dab9688707739e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.7138072360246%2C%20-73.99623982171583%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_7c130c40668347fcb236a7e4265600b8%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_72c6c88fb8984e259263f482abddf801%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_72c6c88fb8984e259263f482abddf801%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EChinatown%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_7c130c40668347fcb236a7e4265600b8.setContent%28html_72c6c88fb8984e259263f482abddf801%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_579e08eb8b964f8789dab9688707739e.bindPopup%28popup_7c130c40668347fcb236a7e4265600b8%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_8a271f5ddb454388b3631e0855058268%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.858515668105774%2C%20-73.88526586246093%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_65c8717fb3e0423993ebc4cabed6d255%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e71e708b301742db959bc3334fa37eb1%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e71e708b301742db959bc3334fa37eb1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBelmont%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_65c8717fb3e0423993ebc4cabed6d255.setContent%28html_e71e708b301742db959bc3334fa37eb1%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_8a271f5ddb454388b3631e0855058268.bindPopup%28popup_65c8717fb3e0423993ebc4cabed6d255%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b084dbe2a9e44c6ea22839fa3eca9bea%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.84431455274919%2C%20-73.88121161953562%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3242808401ae4de9b04999fe12362d49%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1b09885c92544479a1d74e993af955c7%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1b09885c92544479a1d74e993af955c7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEast%20Tremont%2C%20Bronx%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3242808401ae4de9b04999fe12362d49.setContent%28html_1b09885c92544479a1d74e993af955c7%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b084dbe2a9e44c6ea22839fa3eca9bea.bindPopup%28popup_3242808401ae4de9b04999fe12362d49%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e0230d1fcbbd48219ae3207470f909de%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.677032992054556%2C%20-73.91306544107215%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_7e07b99dccd74ca3b5a4bb49e551e4a6%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e90878361b7d420883b0f25b2617ace0%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e90878361b7d420883b0f25b2617ace0%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EOcean%20Hill%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_7e07b99dccd74ca3b5a4bb49e551e4a6.setContent%28html_e90878361b7d420883b0f25b2617ace0%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e0230d1fcbbd48219ae3207470f909de.bindPopup%28popup_7e07b99dccd74ca3b5a4bb49e551e4a6%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_20c0097474c44437a7c0585ad81d2a68%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.66399454264072%2C%20-73.91031024006473%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_54423c770e854fcd9cf7414b916f1de2%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7b4b24497676418ab1f7cba54e610fc4%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7b4b24497676418ab1f7cba54e610fc4%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBrownsville%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_54423c770e854fcd9cf7414b916f1de2.setContent%28html_7b4b24497676418ab1f7cba54e610fc4%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_20c0097474c44437a7c0585ad81d2a68.bindPopup%28popup_54423c770e854fcd9cf7414b916f1de2%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b133bf5cfbdf4861bf738361a39ee6c8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.69732776353425%2C%20-73.98634049601266%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_48268fd4876b4e378b110d62a1eb3283%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9aa1f616bfb74c5baf32a1f525d4fdfd%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9aa1f616bfb74c5baf32a1f525d4fdfd%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDUMBO-Vinegar%20Hill-Downtown%20Brooklyn-Boerum%20Hill%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_48268fd4876b4e378b110d62a1eb3283.setContent%28html_9aa1f616bfb74c5baf32a1f525d4fdfd%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b133bf5cfbdf4861bf738361a39ee6c8.bindPopup%28popup_48268fd4876b4e378b110d62a1eb3283%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2c7f67c0c32e4ec2a6146563de5cb477%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.69188111856425%2C%20-73.97766303689505%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3b1abc35499c44a0bb6cd4c5d100175b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_92204c3b7a9b48a3814a6768c80c4b09%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_92204c3b7a9b48a3814a6768c80c4b09%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFort%20Greene%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3b1abc35499c44a0bb6cd4c5d100175b.setContent%28html_92204c3b7a9b48a3814a6768c80c4b09%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2c7f67c0c32e4ec2a6146563de5cb477.bindPopup%28popup_3b1abc35499c44a0bb6cd4c5d100175b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2ad0e6a83de34580a769bf6b58ee32f1%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.82607099769937%2C%20-73.93699150049123%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b7c59e7660b34cda897d7910d41ad586%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3ccf2b74eb2f4ea69abed50081450a5c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_3ccf2b74eb2f4ea69abed50081450a5c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECentral%20Harlem%20North-Polo%20Grounds%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b7c59e7660b34cda897d7910d41ad586.setContent%28html_3ccf2b74eb2f4ea69abed50081450a5c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2ad0e6a83de34580a769bf6b58ee32f1.bindPopup%28popup_b7c59e7660b34cda897d7910d41ad586%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a466bc683d9e4cec926f3b3a220588e4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.819604017014434%2C%20-73.9561130807665%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e47f26d5337c442981f879e8327b1f20%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a875875a2aa54d028a2f622f70bd18d4%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_a875875a2aa54d028a2f622f70bd18d4%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EManhattanville%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e47f26d5337c442981f879e8327b1f20.setContent%28html_a875875a2aa54d028a2f622f70bd18d4%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a466bc683d9e4cec926f3b3a220588e4.bindPopup%28popup_e47f26d5337c442981f879e8327b1f20%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_24f9fc2f01fc407394824b9a1c80f925%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.65866163110918%2C%20-73.94712520120946%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_9a09246026594e6c8aba94c7e3ec26da%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c26f76df74e94bf69571b3918cd64088%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c26f76df74e94bf69571b3918cd64088%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EProspect%20Lefferts%20Gardens-Wingate%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_9a09246026594e6c8aba94c7e3ec26da.setContent%28html_c26f76df74e94bf69571b3918cd64088%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_24f9fc2f01fc407394824b9a1c80f925.bindPopup%28popup_9a09246026594e6c8aba94c7e3ec26da%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_bbe1ec8cb1604619985a937e172b30c4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.6664526587511%2C%20-73.94677939547148%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c574b79afda74cae89cbfb6aaef25566%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_15fbe29c819c4b979c91a9fd73c0a5a6%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_15fbe29c819c4b979c91a9fd73c0a5a6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECrown%20Heights%20South%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c574b79afda74cae89cbfb6aaef25566.setContent%28html_15fbe29c819c4b979c91a9fd73c0a5a6%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_bbe1ec8cb1604619985a937e172b30c4.bindPopup%28popup_c574b79afda74cae89cbfb6aaef25566%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5d79bfe63bb846488e6beced77271b58%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.655996714424845%2C%20-73.87666420428302%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5eb73392d7b44b34b16fde0f42bce476%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_75cf727fdc19457ea56aa7da11686365%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_75cf727fdc19457ea56aa7da11686365%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEast%20New%20York%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5eb73392d7b44b34b16fde0f42bce476.setContent%28html_75cf727fdc19457ea56aa7da11686365%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_5d79bfe63bb846488e6beced77271b58.bindPopup%28popup_5eb73392d7b44b34b16fde0f42bce476%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2f1de232e885465090b063768ebbbb28%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.647071829703194%2C%20-73.88837630756214%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a4de3d5233a64a7d80ac9033cf699ad5%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_bec5bcdc8395466f9ace3c62c95a22be%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_bec5bcdc8395466f9ace3c62c95a22be%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EStarrett%20City%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a4de3d5233a64a7d80ac9033cf699ad5.setContent%28html_bec5bcdc8395466f9ace3c62c95a22be%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2f1de232e885465090b063768ebbbb28.bindPopup%28popup_a4de3d5233a64a7d80ac9033cf699ad5%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_93acb4a3e4634b2ebae31c1eea232d01%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.71489326776224%2C%20-73.9620634709816%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c2963e3ea9d44b379bfb63c5b7141c86%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_eacb9ad60b9c4854bce2311a60a184db%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_eacb9ad60b9c4854bce2311a60a184db%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENorth%20Side-South%20Side%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c2963e3ea9d44b379bfb63c5b7141c86.setContent%28html_eacb9ad60b9c4854bce2311a60a184db%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_93acb4a3e4634b2ebae31c1eea232d01.bindPopup%28popup_c2963e3ea9d44b379bfb63c5b7141c86%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_172280bc22a648e18d97838887156afd%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.715326355663564%2C%20-73.93085933902276%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ec2dabbfd4f0489eaeebad94cec59f5f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a47dd0accac84f0a9d63a0e27a25edca%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_a47dd0accac84f0a9d63a0e27a25edca%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEast%20Williamsburg%2C%20Brooklyn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ec2dabbfd4f0489eaeebad94cec59f5f.setContent%28html_a47dd0accac84f0a9d63a0e27a25edca%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_172280bc22a648e18d97838887156afd.bindPopup%28popup_ec2dabbfd4f0489eaeebad94cec59f5f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c94d64d7e4a6405ab8601d57d84ee43c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.736507018580056%2C%20-73.98381323573406%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_90989b302c3c4d56a5f2e011841769d4%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e033c1074ae145f9a8f85dfefa6f3457%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e033c1074ae145f9a8f85dfefa6f3457%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGramercy%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_90989b302c3c4d56a5f2e011841769d4.setContent%28html_e033c1074ae145f9a8f85dfefa6f3457%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c94d64d7e4a6405ab8601d57d84ee43c.bindPopup%28popup_90989b302c3c4d56a5f2e011841769d4%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4bab412578b24d8a9c7936a591e257f4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.73168015130651%2C%20-73.97437379671852%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_66672434f13e4454b4cf37ec34df5800%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d8cb9942686f4b8f8a22363b037b8d22%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5e3789f8a1454029885eb8ec76c0fb06%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_5e3789f8a1454029885eb8ec76c0fb06%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EStuyvesant%20Town-Cooper%20Village%2C%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d8cb9942686f4b8f8a22363b037b8d22.setContent%28html_5e3789f8a1454029885eb8ec76c0fb06%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4bab412578b24d8a9c7936a591e257f4.bindPopup%28popup_d8cb9942686f4b8f8a22363b037b8d22%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%3C/script%3E onload="this.contentDocument.open();this.contentDocument.write(    decodeURIComponent(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



# Pre-Processing of the Subway stations dataset


```python
df_subw = pd.read_csv("https://query.data.world/s/rttrjnk7raatdri6ialljpsucvbv5b")
```


```python
df_subw.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>URL</th>
      <th>OBJECTID</th>
      <th>NAME</th>
      <th>the_geom</th>
      <th>LINE</th>
      <th>NOTES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>http://web.mta.info/nyct/service/</td>
      <td>1</td>
      <td>Astor Pl</td>
      <td>POINT (-73.99106999861966 40.73005400028978)</td>
      <td>4-6-6 Express</td>
      <td>4 nights, 6-all times, 6 Express-weekdays AM s...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>http://web.mta.info/nyct/service/</td>
      <td>2</td>
      <td>Canal St</td>
      <td>POINT (-74.00019299927328 40.71880300107709)</td>
      <td>4-6-6 Express</td>
      <td>4 nights, 6-all times, 6 Express-weekdays AM s...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>http://web.mta.info/nyct/service/</td>
      <td>3</td>
      <td>50th St</td>
      <td>POINT (-73.98384899986625 40.76172799961419)</td>
      <td>1-2</td>
      <td>1-all times, 2-nights</td>
    </tr>
    <tr>
      <th>3</th>
      <td>http://web.mta.info/nyct/service/</td>
      <td>4</td>
      <td>Bergen St</td>
      <td>POINT (-73.97499915116808 40.68086213682956)</td>
      <td>2-3-4</td>
      <td>4-nights, 3-all other times, 2-all times</td>
    </tr>
    <tr>
      <th>4</th>
      <td>http://web.mta.info/nyct/service/</td>
      <td>5</td>
      <td>Pennsylvania Ave</td>
      <td>POINT (-73.89488591154061 40.66471445143568)</td>
      <td>3-4</td>
      <td>4-nights, 3-all other times</td>
    </tr>
  </tbody>
</table>
</div>



As you can see on the table above, it required to extract the coordianates of subway stations

- Function to extract coordinates


```python
def coord(x): 
    z = x.replace("POINT", "").replace("(", "").replace(")", "").split(" ")
    return  float(z[1]), float(z[2])
```

- Create a DataFrame of subway stations with their name, longitude and latitude


```python
column_names = ['Name', 'Latitude', 'Longitude'] 
# instantiate the dataframe
subw = pd.DataFrame(columns=column_names)
lonlat = list(map(coord, df_subw["the_geom"] ))
subw["Latitude"]= [l[1] for l in lonlat]
subw["Longitude"]= [l[0] for l in lonlat]
subw["Name"] = df_subw["NAME"]
subw.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Astor Pl</td>
      <td>40.730054</td>
      <td>-73.991070</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Canal St</td>
      <td>40.718803</td>
      <td>-74.000193</td>
    </tr>
    <tr>
      <th>2</th>
      <td>50th St</td>
      <td>40.761728</td>
      <td>-73.983849</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bergen St</td>
      <td>40.680862</td>
      <td>-73.974999</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pennsylvania Ave</td>
      <td>40.664714</td>
      <td>-73.894886</td>
    </tr>
  </tbody>
</table>
</div>



- Algorithm used to calculate the distance between two points with their coordinates in decimal degrees


```python
from math import *
def Distance(lat1, lon1, lat2, lon2):
    l1 = radians(lat1);
    l2 = radians(lat2);
    dlat = radians(lat2-lat1);
    dlon = radians(lon2-lon1);
    a = sin(dlat/2) * sin(dlat/2) + cos(l1) * cos(l2) * sin(dlon/2) * sin(dlon/2);
    c = 2 * atan2(sqrt(a), sqrt(1-a));
    d = 6371 * c # in km
    return d
```

* Here, we determine the distance from each district to the nearest subway, and the number of subways with a minimal distance of 1.5km


```python
res, nb = [], []
for lat, lon in zip(df['Latitude'], df['Longitude']):
    z = []
    k = 0
    for lat1, lon1 in zip(subw['Latitude'], subw['Longitude']):
        dist = Distance(lat1, lon1, lat, lon)
        if dist<1.5: k = k+1
        z.append(dist)
    res.append(min(z))
    nb.append(k)
```


```python
df["min_dist_to_subways_km"] = res
```


```python
df["number_subways_1.5km"] = nb
```


```python
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>DistrictCode</th>
      <th>DistrictName</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Population</th>
      <th>min_dist_to_subways_km</th>
      <th>number_subways_1.5km</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Manhattan</td>
      <td>MN15</td>
      <td>Clinton</td>
      <td>40.766378</td>
      <td>-73.996548</td>
      <td>45884</td>
      <td>0.990834</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Manhattan</td>
      <td>MN25</td>
      <td>Battery Park City-Lower Manhattan</td>
      <td>40.694739</td>
      <td>-74.001444</td>
      <td>39699</td>
      <td>0.767109</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Manhattan</td>
      <td>MN14</td>
      <td>Lincoln Square</td>
      <td>40.775236</td>
      <td>-73.988364</td>
      <td>61489</td>
      <td>0.555412</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Manhattan</td>
      <td>MN17</td>
      <td>Midtown-Midtown South</td>
      <td>40.756759</td>
      <td>-73.982858</td>
      <td>28630</td>
      <td>0.245619</td>
      <td>32</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Manhattan</td>
      <td>MN40</td>
      <td>Upper East Side-Carnegie Hill</td>
      <td>40.775283</td>
      <td>-73.960816</td>
      <td>61207</td>
      <td>0.201265</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



## Using the Foursquare API to retrieve information of the venues including the restaurants in  New York NTA (districts). The API will return a JSON file which will be further converted into a Python Dataframe.


```python
CLIENT_ID = 'NPXEJNZYDELVG0KRFPVPZ3AAUJFKPJLIGVKAXN1MZ0VLKT40' # your Foursquare ID
CLIENT_SECRET = '5Q1BDASO43ZB5AQZVEJLE3UJDSK4SPDXXONMYC201MRDWIXV' # your Foursquare Secret
ACCESS_TOKEN = 'J3B4AQ5DAY2G4PENKGKEJM0XW1P3QWREURLU3IYWMWJLOUBO' # your FourSquare Access Token
VERSION = '20180604'
LIMIT = 100
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
```

    Your credentails:
    CLIENT_ID: NPXEJNZYDELVG0KRFPVPZ3AAUJFKPJLIGVKAXN1MZ0VLKT40
    CLIENT_SECRET:5Q1BDASO43ZB5AQZVEJLE3UJDSK4SPDXXONMYC201MRDWIXV
    

* We select only Manhattan borough to restrict our analysis borough by borough. An other could be selected.


```python
manhattan_data = df[df['Borough'] == 'Manhattan'].reset_index(drop=True)
manhattan_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>DistrictCode</th>
      <th>DistrictName</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Population</th>
      <th>min_dist_to_subways_km</th>
      <th>number_subways_1.5km</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Manhattan</td>
      <td>MN15</td>
      <td>Clinton</td>
      <td>40.766378</td>
      <td>-73.996548</td>
      <td>45884</td>
      <td>0.990834</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Manhattan</td>
      <td>MN25</td>
      <td>Battery Park City-Lower Manhattan</td>
      <td>40.694739</td>
      <td>-74.001444</td>
      <td>39699</td>
      <td>0.767109</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Manhattan</td>
      <td>MN14</td>
      <td>Lincoln Square</td>
      <td>40.775236</td>
      <td>-73.988364</td>
      <td>61489</td>
      <td>0.555412</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Manhattan</td>
      <td>MN17</td>
      <td>Midtown-Midtown South</td>
      <td>40.756759</td>
      <td>-73.982858</td>
      <td>28630</td>
      <td>0.245619</td>
      <td>32</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Manhattan</td>
      <td>MN40</td>
      <td>Upper East Side-Carnegie Hill</td>
      <td>40.775283</td>
      <td>-73.960816</td>
      <td>61207</td>
      <td>0.201265</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
address = 'Manhattan, NY'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Manhattan are {}, {}.'.format(latitude, longitude))
```

    The geograpical coordinate of Manhattan are 40.7896239, -73.9598939.
    

- create map of Manhattan using latitude and longitude values


```python
map_manhattan = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(manhattan_data['Latitude'], manhattan_data['Longitude'], manhattan_data['DistrictName']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_manhattan)  
    
map_manhattan
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=%3C%21DOCTYPE%20html%3E%0A%3Chead%3E%20%20%20%20%0A%20%20%20%20%3Cmeta%20http-equiv%3D%22content-type%22%20content%3D%22text/html%3B%20charset%3DUTF-8%22%20/%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%3Cscript%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20L_NO_TOUCH%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20L_DISABLE_3D%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%3C/script%3E%0A%20%20%20%20%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//code.jquery.com/jquery-1.12.4.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js%22%3E%3C/script%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//rawcdn.githack.com/python-visualization/folium/master/folium/templates/leaflet.awesome.rotate.css%22/%3E%0A%20%20%20%20%3Cstyle%3Ehtml%2C%20body%20%7Bwidth%3A%20100%25%3Bheight%3A%20100%25%3Bmargin%3A%200%3Bpadding%3A%200%3B%7D%3C/style%3E%0A%20%20%20%20%3Cstyle%3E%23map%20%7Bposition%3Aabsolute%3Btop%3A0%3Bbottom%3A0%3Bright%3A0%3Bleft%3A0%3B%7D%3C/style%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cmeta%20name%3D%22viewport%22%20content%3D%22width%3Ddevice-width%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20initial-scale%3D1.0%2C%20maximum-scale%3D1.0%2C%20user-scalable%3Dno%22%20/%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cstyle%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%23map_f23774c8082b4fdf8d2be6798c657cc8%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20position%3A%20relative%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20width%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20height%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20left%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20top%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%3C/style%3E%0A%20%20%20%20%20%20%20%20%0A%3C/head%3E%0A%3Cbody%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cdiv%20class%3D%22folium-map%22%20id%3D%22map_f23774c8082b4fdf8d2be6798c657cc8%22%20%3E%3C/div%3E%0A%20%20%20%20%20%20%20%20%0A%3C/body%3E%0A%3Cscript%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20map_f23774c8082b4fdf8d2be6798c657cc8%20%3D%20L.map%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22map_f23774c8082b4fdf8d2be6798c657cc8%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20center%3A%20%5B40.7896239%2C%20-73.9598939%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20crs%3A%20L.CRS.EPSG3857%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoom%3A%2011%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoomControl%3A%20true%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20preferCanvas%3A%20false%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20tile_layer_16e84a398daf40d4b74200f8692dc68a%20%3D%20L.tileLayer%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22https%3A//%7Bs%7D.tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22attribution%22%3A%20%22Data%20by%20%5Cu0026copy%3B%20%5Cu003ca%20href%3D%5C%22http%3A//openstreetmap.org%5C%22%5Cu003eOpenStreetMap%5Cu003c/a%5Cu003e%2C%20under%20%5Cu003ca%20href%3D%5C%22http%3A//www.openstreetmap.org/copyright%5C%22%5Cu003eODbL%5Cu003c/a%5Cu003e.%22%2C%20%22detectRetina%22%3A%20false%2C%20%22maxNativeZoom%22%3A%2018%2C%20%22maxZoom%22%3A%2018%2C%20%22minZoom%22%3A%200%2C%20%22noWrap%22%3A%20false%2C%20%22opacity%22%3A%201%2C%20%22subdomains%22%3A%20%22abc%22%2C%20%22tms%22%3A%20false%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_faa934a475f444efbb7a879052a000d7%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.7663775596076%2C%20-73.99654837674895%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_718c07f0dde94d4c804a8507a6068e7e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_358d3661a7c34bc5882e83ac49ab4527%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_358d3661a7c34bc5882e83ac49ab4527%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EClinton%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_718c07f0dde94d4c804a8507a6068e7e.setContent%28html_358d3661a7c34bc5882e83ac49ab4527%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_faa934a475f444efbb7a879052a000d7.bindPopup%28popup_718c07f0dde94d4c804a8507a6068e7e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_cfc2532307cc41d98a20d4b3bf7d6522%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.69473916696892%2C%20-74.00144418241418%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_97126cfcc3514ff0942d90f83ece9dcf%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_554d867769eb4c76b45057f3bd8d524a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_554d867769eb4c76b45057f3bd8d524a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBattery%20Park%20City-Lower%20Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_97126cfcc3514ff0942d90f83ece9dcf.setContent%28html_554d867769eb4c76b45057f3bd8d524a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_cfc2532307cc41d98a20d4b3bf7d6522.bindPopup%28popup_97126cfcc3514ff0942d90f83ece9dcf%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3fddf1ebc4034de3a886fa46e3b7e918%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.77523608214009%2C%20-73.98836366766845%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3f63a0b3f9a2423c97128297f63516c7%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5bb7d7ef268f49678d695f2c5029d596%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_5bb7d7ef268f49678d695f2c5029d596%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELincoln%20Square%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3f63a0b3f9a2423c97128297f63516c7.setContent%28html_5bb7d7ef268f49678d695f2c5029d596%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_3fddf1ebc4034de3a886fa46e3b7e918.bindPopup%28popup_3f63a0b3f9a2423c97128297f63516c7%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_19b61332b97643d6915ac5634e63932d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.75675923700698%2C%20-73.98285820774203%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_9a6c47a66fcc481084b0b276394e70b6%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_40a278b490bd4b4cbb49fc323105839e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_40a278b490bd4b4cbb49fc323105839e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMidtown-Midtown%20South%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_9a6c47a66fcc481084b0b276394e70b6.setContent%28html_40a278b490bd4b4cbb49fc323105839e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_19b61332b97643d6915ac5634e63932d.bindPopup%28popup_9a6c47a66fcc481084b0b276394e70b6%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_31dd9abd5d19406c930e6245bc1b173c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.77528339839104%2C%20-73.96081636591553%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_31c88eba5f974952ae2ce24e67ce2de7%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7919a5110a2641e6aa5d8d8415e99684%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7919a5110a2641e6aa5d8d8415e99684%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EUpper%20East%20Side-Carnegie%20Hill%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_31c88eba5f974952ae2ce24e67ce2de7.setContent%28html_7919a5110a2641e6aa5d8d8415e99684%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_31dd9abd5d19406c930e6245bc1b173c.bindPopup%28popup_31c88eba5f974952ae2ce24e67ce2de7%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_25081ee224a3447aaafb914178c05062%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.78050399554715%2C%20-73.93852233514457%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_cc1c6fae4aef49ab8b19b0a9c68dee8d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8c7f9328c0f244c68893713c66325b92%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8c7f9328c0f244c68893713c66325b92%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EYorkville%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_cc1c6fae4aef49ab8b19b0a9c68dee8d.setContent%28html_8c7f9328c0f244c68893713c66325b92%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_25081ee224a3447aaafb914178c05062.bindPopup%28popup_cc1c6fae4aef49ab8b19b0a9c68dee8d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b7ef7b4b8c8f4a11b26f570e243085b3%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.85601276666469%2C%20-73.92208086039953%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_fe9113e004544a36a446ac320657c4f2%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_304e1b8896e2499a93b9ad85bad5cd32%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_304e1b8896e2499a93b9ad85bad5cd32%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMarble%20Hill-Inwood%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_fe9113e004544a36a446ac320657c4f2.setContent%28html_304e1b8896e2499a93b9ad85bad5cd32%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b7ef7b4b8c8f4a11b26f570e243085b3.bindPopup%28popup_fe9113e004544a36a446ac320657c4f2%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6cc7efd9f1534d99a9d0c57eb4d270ff%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.74911729284833%2C%20-74.00593919513614%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_9e23c045bab948ecbf0461395768b089%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9bcc55bb4a3b4514b13ff17ab7666893%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9bcc55bb4a3b4514b13ff17ab7666893%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHudson%20Yards-Chelsea-Flatiron-Union%20Square%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_9e23c045bab948ecbf0461395768b089.setContent%28html_9bcc55bb4a3b4514b13ff17ab7666893%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_6cc7efd9f1534d99a9d0c57eb4d270ff.bindPopup%28popup_9e23c045bab948ecbf0461395768b089%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9fbda2cad7cd4f06aa6a7e669ef53fe8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.734565184788664%2C%20-74.00621066209094%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_9963a0bf66544d9f9d9ebf91529d48f3%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8d8460dc57e64a5dbee65f03a60c8369%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8d8460dc57e64a5dbee65f03a60c8369%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWest%20Village%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_9963a0bf66544d9f9d9ebf91529d48f3.setContent%28html_8d8460dc57e64a5dbee65f03a60c8369%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9fbda2cad7cd4f06aa6a7e669ef53fe8.bindPopup%28popup_9963a0bf66544d9f9d9ebf91529d48f3%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_779d6e4efc3b4b94b9259f3bcb044a76%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.79083355054379%2C%20-73.97692336107843%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1f0fbcd7d6d84f36ad5280d21cca1356%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_457f141fd6eb45ccbcffdbcc7f2c2de1%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_457f141fd6eb45ccbcffdbcc7f2c2de1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EUpper%20West%20Side%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1f0fbcd7d6d84f36ad5280d21cca1356.setContent%28html_457f141fd6eb45ccbcffdbcc7f2c2de1%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_779d6e4efc3b4b94b9259f3bcb044a76.bindPopup%28popup_1f0fbcd7d6d84f36ad5280d21cca1356%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_bc09b914fe804da2b01dda0374d01a69%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.827993805267916%2C%20-73.95054568805168%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5055defa5a11449699594a3ff0101b7e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6fead1deb3404f269f823b5c2af2ea31%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_6fead1deb3404f269f823b5c2af2ea31%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHamilton%20Heights%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5055defa5a11449699594a3ff0101b7e.setContent%28html_6fead1deb3404f269f823b5c2af2ea31%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_bc09b914fe804da2b01dda0374d01a69.bindPopup%28popup_5055defa5a11449699594a3ff0101b7e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_78632e7788d14f38b979feb45f8ec3c3%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.78819344298152%2C%20-73.9423712473909%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_2d59892a226f4bb9a54bc7bc80994996%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2de94cce886041289020c067fc80dac4%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2de94cce886041289020c067fc80dac4%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEast%20Harlem%20South%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_2d59892a226f4bb9a54bc7bc80994996.setContent%28html_2de94cce886041289020c067fc80dac4%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_78632e7788d14f38b979feb45f8ec3c3.bindPopup%28popup_2d59892a226f4bb9a54bc7bc80994996%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_de59b8b21b124ecab34aa3ec2917a290%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.68548712038733%2C%20-74.0095551305848%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_0bf5f888f5e045de9aefdb6eeb920789%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_22e241bdec6c4dec98df11d6403a51df%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_22e241bdec6c4dec98df11d6403a51df%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3Epark-cemetery-etc-Manhattan%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_0bf5f888f5e045de9aefdb6eeb920789.setContent%28html_22e241bdec6c4dec98df11d6403a51df%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_de59b8b21b124ecab34aa3ec2917a290.bindPopup%28popup_0bf5f888f5e045de9aefdb6eeb920789%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2e27d3e2b4ca41bc9206f17ef93f79bb%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.80347835059119%2C%20-73.95096528365056%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_cde2b29b0a154ecfa17769e7ecef2b99%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2ac6785c862a497ba9f17b343c5d33f4%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2ac6785c862a497ba9f17b343c5d33f4%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECentral%20Harlem%20South%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_cde2b29b0a154ecfa17769e7ecef2b99.setContent%28html_2ac6785c862a497ba9f17b343c5d33f4%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2e27d3e2b4ca41bc9206f17ef93f79bb.bindPopup%28popup_cde2b29b0a154ecfa17769e7ecef2b99%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_bcc937b19ce649e4831e075f391c8fd5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.80515483757544%2C%20-73.93494660371782%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_61056f140bc14d5bbb786f31e02c59c7%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b3b2e25e6ac84d5b8f946941b608710a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b3b2e25e6ac84d5b8f946941b608710a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEast%20Harlem%20North%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_61056f140bc14d5bbb786f31e02c59c7.setContent%28html_b3b2e25e6ac84d5b8f946941b608710a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_bcc937b19ce649e4831e075f391c8fd5.bindPopup%28popup_61056f140bc14d5bbb786f31e02c59c7%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0b6bc963716049bc960710da7a7005ba%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.72734350120654%2C%20-73.98635207212405%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_19b2e40b1b3448daa05e7aa10790bf06%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6ed311ae031145ea8d873f69c7dc17ef%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_6ed311ae031145ea8d873f69c7dc17ef%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEast%20Village%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_19b2e40b1b3448daa05e7aa10790bf06.setContent%28html_6ed311ae031145ea8d873f69c7dc17ef%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_0b6bc963716049bc960710da7a7005ba.bindPopup%28popup_19b2e40b1b3448daa05e7aa10790bf06%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_85f7216eba4f4fbdbc650454081c84ac%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.705687831163246%2C%20-73.98203911644805%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6d79e29ec3dd4dc49b22ce9cca9d0215%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ecdb6d5ef4e4490aad0643f95567e6e9%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ecdb6d5ef4e4490aad0643f95567e6e9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELower%20East%20Side%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6d79e29ec3dd4dc49b22ce9cca9d0215.setContent%28html_ecdb6d5ef4e4490aad0643f95567e6e9%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_85f7216eba4f4fbdbc650454081c84ac.bindPopup%28popup_6d79e29ec3dd4dc49b22ce9cca9d0215%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d8e95158545b47ccb74f8d48d31aa0a5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.73982212741276%2C%20-73.97474305311528%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_652cd5ba18d3448988adcefbffef0a0b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ae90f36bfaa54dbfb3f166f0946aade1%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ae90f36bfaa54dbfb3f166f0946aade1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMurray%20Hill-Kips%20Bay%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_652cd5ba18d3448988adcefbffef0a0b.setContent%28html_ae90f36bfaa54dbfb3f166f0946aade1%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_d8e95158545b47ccb74f8d48d31aa0a5.bindPopup%28popup_652cd5ba18d3448988adcefbffef0a0b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_694f7b2d0e15489c8a3f009b3b068bd1%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.808454248987175%2C%20-73.96406710398044%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_67a0c8cb541f43ef9cdd78ee373c0e97%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_57f2e86744fe4e94a8cbb37d174ea5e8%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_57f2e86744fe4e94a8cbb37d174ea5e8%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMorningside%20Heights%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_67a0c8cb541f43ef9cdd78ee373c0e97.setContent%28html_57f2e86744fe4e94a8cbb37d174ea5e8%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_694f7b2d0e15489c8a3f009b3b068bd1.bindPopup%28popup_67a0c8cb541f43ef9cdd78ee373c0e97%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_047241fbc9904d06987679ded0a9fa76%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.85765544136461%2C%20-73.93734644214263%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3101a2400edf4f9abc97b317bac4a54e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_bf35b9ea9ca14d6a92be6800b8af155a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_bf35b9ea9ca14d6a92be6800b8af155a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWashington%20Heights%20North%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3101a2400edf4f9abc97b317bac4a54e.setContent%28html_bf35b9ea9ca14d6a92be6800b8af155a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_047241fbc9904d06987679ded0a9fa76.bindPopup%28popup_3101a2400edf4f9abc97b317bac4a54e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f745fb28300f4928b278faecd4ef0e93%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.84274234313284%2C%20-73.9433111501022%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_33ad8916314f4c488d79665fb6f2633c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9b8f241b17f9437eb20744311a801061%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9b8f241b17f9437eb20744311a801061%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWashington%20Heights%20South%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_33ad8916314f4c488d79665fb6f2633c.setContent%28html_9b8f241b17f9437eb20744311a801061%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f745fb28300f4928b278faecd4ef0e93.bindPopup%28popup_33ad8916314f4c488d79665fb6f2633c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d8ef9929d22b418f8ca67d8400d377e9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.74880110296486%2C%20-73.96949297606015%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_03cf6287178c4574a90d71d5515de851%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7d493dd0b255477b93f2b8b8dbe7903b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7d493dd0b255477b93f2b8b8dbe7903b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ETurtle%20Bay-East%20Midtown%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_03cf6287178c4574a90d71d5515de851.setContent%28html_7d493dd0b255477b93f2b8b8dbe7903b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_d8ef9929d22b418f8ca67d8400d377e9.bindPopup%28popup_03cf6287178c4574a90d71d5515de851%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f613b619c499486d963d1767b796c41c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.748395140173336%2C%20-73.95764383283307%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_36282dc033504585922d82ea871dd5ca%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2cf33d4610f74e0bbf2087cca05c5355%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2cf33d4610f74e0bbf2087cca05c5355%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELenox%20Hill-Roosevelt%20Island%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_36282dc033504585922d82ea871dd5ca.setContent%28html_2cf33d4610f74e0bbf2087cca05c5355%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f613b619c499486d963d1767b796c41c.bindPopup%28popup_36282dc033504585922d82ea871dd5ca%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f9df9c766c5f4be5a7bb06e3c84bca13%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.72044868765068%2C%20-74.0071853926249%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6f724c48a3cc407b9dfe07c836c1d1aa%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d91729dad2f34b7895ffc6d00c63a068%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d91729dad2f34b7895ffc6d00c63a068%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESoHo-TriBeCa-Civic%20Center-Little%20Italy%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6f724c48a3cc407b9dfe07c836c1d1aa.setContent%28html_d91729dad2f34b7895ffc6d00c63a068%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f9df9c766c5f4be5a7bb06e3c84bca13.bindPopup%28popup_6f724c48a3cc407b9dfe07c836c1d1aa%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e7fa39cb84314aca9a210b18538e1a2e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.7138072360246%2C%20-73.99623982171583%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_65ed3ae7fac94356b6db0c53222a59d3%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2fb508bf57b4430f9c97c0cd8b72963f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2fb508bf57b4430f9c97c0cd8b72963f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EChinatown%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_65ed3ae7fac94356b6db0c53222a59d3.setContent%28html_2fb508bf57b4430f9c97c0cd8b72963f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e7fa39cb84314aca9a210b18538e1a2e.bindPopup%28popup_65ed3ae7fac94356b6db0c53222a59d3%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7d5f808e29784f448e68d3e065856e0a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.82607099769937%2C%20-73.93699150049123%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5e7dafacc7124d7688aff9488a05c209%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e88ba6b752b4456682f1fd2de5176cf9%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e88ba6b752b4456682f1fd2de5176cf9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECentral%20Harlem%20North-Polo%20Grounds%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5e7dafacc7124d7688aff9488a05c209.setContent%28html_e88ba6b752b4456682f1fd2de5176cf9%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7d5f808e29784f448e68d3e065856e0a.bindPopup%28popup_5e7dafacc7124d7688aff9488a05c209%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_cd3939744d854f6a83bb48a9615fd1d1%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.819604017014434%2C%20-73.9561130807665%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_742f7647b5e74166a13f90a5b4236495%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_cfc56feb212b4dbba4ebcff3faaef9ae%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_cfc56feb212b4dbba4ebcff3faaef9ae%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EManhattanville%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_742f7647b5e74166a13f90a5b4236495.setContent%28html_cfc56feb212b4dbba4ebcff3faaef9ae%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_cd3939744d854f6a83bb48a9615fd1d1.bindPopup%28popup_742f7647b5e74166a13f90a5b4236495%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_da635a5bf3fb434e92b23fd31c0674ed%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.736507018580056%2C%20-73.98381323573406%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a04e00415710400da10082ec4fbeaee1%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d1f3d7beb7e34fb8b59b56ac180f90fc%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d1f3d7beb7e34fb8b59b56ac180f90fc%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGramercy%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a04e00415710400da10082ec4fbeaee1.setContent%28html_d1f3d7beb7e34fb8b59b56ac180f90fc%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_da635a5bf3fb434e92b23fd31c0674ed.bindPopup%28popup_a04e00415710400da10082ec4fbeaee1%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d57037548552488fafd51fadae0326fc%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.73168015130651%2C%20-73.97437379671852%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22blue%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%233186cc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_f23774c8082b4fdf8d2be6798c657cc8%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_41e12909f52745749f8a721cbc196db8%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0283827044aa4f039ebdf8aa92f2f177%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_0283827044aa4f039ebdf8aa92f2f177%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EStuyvesant%20Town-Cooper%20Village%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_41e12909f52745749f8a721cbc196db8.setContent%28html_0283827044aa4f039ebdf8aa92f2f177%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_d57037548552488fafd51fadae0326fc.bindPopup%28popup_41e12909f52745749f8a721cbc196db8%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%3C/script%3E onload="this.contentDocument.open();this.contentDocument.write(    decodeURIComponent(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python
manhattan_data.loc[0, 'DistrictName']
```




    'Clinton'




```python
district_latitude = manhattan_data.loc[0, 'Latitude'] # district latitude value
district_longitude = manhattan_data.loc[0, 'Longitude'] # district longitude value

district_name = manhattan_data.loc[0, 'DistrictName'] # district name

print('Latitude and longitude values of {} are {}, {}.'.format(district_name, 
                                                               district_latitude, 
                                                               district_longitude))
```

    Latitude and longitude values of Clinton are 40.7663775596076, -73.99654837674895.
    

- Webscraping of a district data in Manhattan using Foursquare API


```python
LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 500 # define radius
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    district_latitude, 
    district_longitude, 
    radius, 
    LIMIT)
url 

```




    'https://api.foursquare.com/v2/venues/explore?&client_id=NPXEJNZYDELVG0KRFPVPZ3AAUJFKPJLIGVKAXN1MZ0VLKT40&client_secret=5Q1BDASO43ZB5AQZVEJLE3UJDSK4SPDXXONMYC201MRDWIXV&v=20180604&ll=40.7663775596076,-73.99654837674895&radius=500&limit=100'




```python
results = requests.get(url).json()
#results
```


```python
# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']
```

We can observe before scrapping the list of venue in the first district of Manhattan


```python
venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>categories</th>
      <th>lat</th>
      <th>lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ink 48 Hotel</td>
      <td>Hotel</td>
      <td>40.764505</td>
      <td>-73.995987</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Press Lounge</td>
      <td>Hotel Bar</td>
      <td>40.764531</td>
      <td>-73.996029</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Intrepid Sea, Air &amp; Space Museum</td>
      <td>History Museum</td>
      <td>40.764514</td>
      <td>-73.999385</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Intrepid Museum Store</td>
      <td>Gift Shop</td>
      <td>40.764492</td>
      <td>-73.999237</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Print</td>
      <td>American Restaurant</td>
      <td>40.764658</td>
      <td>-73.995808</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))
```

    100 venues were returned by Foursquare.
    

* Generalizing precedent algorithm for all the districts of a borough


```python
def getNearbyVenues(names, latitudes, longitudes, radius=1000):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        #print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['DistrictName', 
                  'District Latitude', 
                  'District Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
```


```python
manhattan_venues = getNearbyVenues(names=manhattan_data['DistrictName'],
                                   latitudes=manhattan_data['Latitude'],
                                   longitudes=manhattan_data['Longitude']
                                  )
```


```python
manhattan_venues.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DistrictName</th>
      <th>District Latitude</th>
      <th>District Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Clinton</td>
      <td>40.766378</td>
      <td>-73.996548</td>
      <td>Ink 48 Hotel</td>
      <td>40.764505</td>
      <td>-73.995987</td>
      <td>Hotel</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Clinton</td>
      <td>40.766378</td>
      <td>-73.996548</td>
      <td>The Press Lounge</td>
      <td>40.764531</td>
      <td>-73.996029</td>
      <td>Hotel Bar</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Clinton</td>
      <td>40.766378</td>
      <td>-73.996548</td>
      <td>Intrepid Sea, Air &amp; Space Museum</td>
      <td>40.764514</td>
      <td>-73.999385</td>
      <td>History Museum</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Clinton</td>
      <td>40.766378</td>
      <td>-73.996548</td>
      <td>Intrepid Museum Store</td>
      <td>40.764492</td>
      <td>-73.999237</td>
      <td>Gift Shop</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Clinton</td>
      <td>40.766378</td>
      <td>-73.996548</td>
      <td>Sullivan Street Bakery</td>
      <td>40.763512</td>
      <td>-73.994837</td>
      <td>Bakery</td>
    </tr>
  </tbody>
</table>
</div>




```python
manhattan_venues_cc=manhattan_venues['Venue Category'].value_counts().to_frame()
```


```python
manhattan_venues_cc.rename(columns={'Venue Category':'Count'})
##

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Coffee Shop</th>
      <td>113</td>
    </tr>
    <tr>
      <th>Park</th>
      <td>98</td>
    </tr>
    <tr>
      <th>Italian Restaurant</th>
      <td>82</td>
    </tr>
    <tr>
      <th>Café</th>
      <td>81</td>
    </tr>
    <tr>
      <th>Pizza Place</th>
      <td>78</td>
    </tr>
    <tr>
      <th>Bakery</th>
      <td>64</td>
    </tr>
    <tr>
      <th>Mexican Restaurant</th>
      <td>63</td>
    </tr>
    <tr>
      <th>Bar</th>
      <td>62</td>
    </tr>
    <tr>
      <th>American Restaurant</th>
      <td>56</td>
    </tr>
    <tr>
      <th>Art Gallery</th>
      <td>47</td>
    </tr>
    <tr>
      <th>Gym</th>
      <td>44</td>
    </tr>
    <tr>
      <th>Wine Shop</th>
      <td>42</td>
    </tr>
    <tr>
      <th>Deli / Bodega</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Cocktail Bar</th>
      <td>38</td>
    </tr>
    <tr>
      <th>Gym / Fitness Center</th>
      <td>38</td>
    </tr>
    <tr>
      <th>Wine Bar</th>
      <td>38</td>
    </tr>
    <tr>
      <th>Theater</th>
      <td>37</td>
    </tr>
    <tr>
      <th>Sandwich Place</th>
      <td>34</td>
    </tr>
    <tr>
      <th>Thai Restaurant</th>
      <td>32</td>
    </tr>
    <tr>
      <th>Grocery Store</th>
      <td>32</td>
    </tr>
    <tr>
      <th>Ice Cream Shop</th>
      <td>32</td>
    </tr>
    <tr>
      <th>Seafood Restaurant</th>
      <td>31</td>
    </tr>
    <tr>
      <th>Chinese Restaurant</th>
      <td>31</td>
    </tr>
    <tr>
      <th>Indian Restaurant</th>
      <td>30</td>
    </tr>
    <tr>
      <th>Japanese Restaurant</th>
      <td>29</td>
    </tr>
    <tr>
      <th>French Restaurant</th>
      <td>29</td>
    </tr>
    <tr>
      <th>Sushi Restaurant</th>
      <td>29</td>
    </tr>
    <tr>
      <th>Latin American Restaurant</th>
      <td>28</td>
    </tr>
    <tr>
      <th>Burger Joint</th>
      <td>28</td>
    </tr>
    <tr>
      <th>Yoga Studio</th>
      <td>28</td>
    </tr>
    <tr>
      <th>Playground</th>
      <td>28</td>
    </tr>
    <tr>
      <th>Hotel</th>
      <td>28</td>
    </tr>
    <tr>
      <th>Dog Run</th>
      <td>25</td>
    </tr>
    <tr>
      <th>Spa</th>
      <td>25</td>
    </tr>
    <tr>
      <th>Bookstore</th>
      <td>24</td>
    </tr>
    <tr>
      <th>New American Restaurant</th>
      <td>24</td>
    </tr>
    <tr>
      <th>Juice Bar</th>
      <td>24</td>
    </tr>
    <tr>
      <th>Bagel Shop</th>
      <td>24</td>
    </tr>
    <tr>
      <th>Scenic Lookout</th>
      <td>22</td>
    </tr>
    <tr>
      <th>Food Truck</th>
      <td>20</td>
    </tr>
    <tr>
      <th>Lounge</th>
      <td>19</td>
    </tr>
    <tr>
      <th>Cosmetics Shop</th>
      <td>18</td>
    </tr>
    <tr>
      <th>Mediterranean Restaurant</th>
      <td>17</td>
    </tr>
    <tr>
      <th>Restaurant</th>
      <td>17</td>
    </tr>
    <tr>
      <th>Dessert Shop</th>
      <td>16</td>
    </tr>
    <tr>
      <th>Caribbean Restaurant</th>
      <td>16</td>
    </tr>
    <tr>
      <th>Salon / Barbershop</th>
      <td>16</td>
    </tr>
    <tr>
      <th>Plaza</th>
      <td>16</td>
    </tr>
    <tr>
      <th>Vegetarian / Vegan Restaurant</th>
      <td>15</td>
    </tr>
    <tr>
      <th>Spanish Restaurant</th>
      <td>14</td>
    </tr>
    <tr>
      <th>Pub</th>
      <td>14</td>
    </tr>
    <tr>
      <th>Garden</th>
      <td>14</td>
    </tr>
    <tr>
      <th>Salad Place</th>
      <td>13</td>
    </tr>
    <tr>
      <th>Clothing Store</th>
      <td>13</td>
    </tr>
    <tr>
      <th>Gourmet Shop</th>
      <td>13</td>
    </tr>
    <tr>
      <th>Asian Restaurant</th>
      <td>13</td>
    </tr>
    <tr>
      <th>Speakeasy</th>
      <td>12</td>
    </tr>
    <tr>
      <th>Tapas Restaurant</th>
      <td>12</td>
    </tr>
    <tr>
      <th>Southern / Soul Food Restaurant</th>
      <td>12</td>
    </tr>
    <tr>
      <th>Vietnamese Restaurant</th>
      <td>12</td>
    </tr>
    <tr>
      <th>Steakhouse</th>
      <td>12</td>
    </tr>
    <tr>
      <th>Taco Place</th>
      <td>11</td>
    </tr>
    <tr>
      <th>Dance Studio</th>
      <td>11</td>
    </tr>
    <tr>
      <th>Bank</th>
      <td>11</td>
    </tr>
    <tr>
      <th>Greek Restaurant</th>
      <td>11</td>
    </tr>
    <tr>
      <th>Exhibit</th>
      <td>11</td>
    </tr>
    <tr>
      <th>Korean Restaurant</th>
      <td>10</td>
    </tr>
    <tr>
      <th>Jazz Club</th>
      <td>10</td>
    </tr>
    <tr>
      <th>Pier</th>
      <td>10</td>
    </tr>
    <tr>
      <th>Farmers Market</th>
      <td>10</td>
    </tr>
    <tr>
      <th>Gastropub</th>
      <td>10</td>
    </tr>
    <tr>
      <th>Diner</th>
      <td>10</td>
    </tr>
    <tr>
      <th>Track</th>
      <td>10</td>
    </tr>
    <tr>
      <th>Gift Shop</th>
      <td>9</td>
    </tr>
    <tr>
      <th>Ramen Restaurant</th>
      <td>9</td>
    </tr>
    <tr>
      <th>Pharmacy</th>
      <td>9</td>
    </tr>
    <tr>
      <th>Tennis Court</th>
      <td>9</td>
    </tr>
    <tr>
      <th>Pool</th>
      <td>9</td>
    </tr>
    <tr>
      <th>Supermarket</th>
      <td>9</td>
    </tr>
    <tr>
      <th>Cycle Studio</th>
      <td>9</td>
    </tr>
    <tr>
      <th>Furniture / Home Store</th>
      <td>9</td>
    </tr>
    <tr>
      <th>Trail</th>
      <td>8</td>
    </tr>
    <tr>
      <th>Pet Store</th>
      <td>8</td>
    </tr>
    <tr>
      <th>Boutique</th>
      <td>8</td>
    </tr>
    <tr>
      <th>Middle Eastern Restaurant</th>
      <td>8</td>
    </tr>
    <tr>
      <th>Historic Site</th>
      <td>8</td>
    </tr>
    <tr>
      <th>Beer Bar</th>
      <td>8</td>
    </tr>
    <tr>
      <th>Performing Arts Venue</th>
      <td>8</td>
    </tr>
    <tr>
      <th>Art Museum</th>
      <td>8</td>
    </tr>
    <tr>
      <th>Donut Shop</th>
      <td>8</td>
    </tr>
    <tr>
      <th>Indie Theater</th>
      <td>8</td>
    </tr>
    <tr>
      <th>Cheese Shop</th>
      <td>8</td>
    </tr>
    <tr>
      <th>Arts &amp; Crafts Store</th>
      <td>7</td>
    </tr>
    <tr>
      <th>Food &amp; Drink Shop</th>
      <td>7</td>
    </tr>
    <tr>
      <th>Mobile Phone Shop</th>
      <td>7</td>
    </tr>
    <tr>
      <th>Men's Store</th>
      <td>7</td>
    </tr>
    <tr>
      <th>Optical Shop</th>
      <td>7</td>
    </tr>
    <tr>
      <th>Outdoor Sculpture</th>
      <td>7</td>
    </tr>
    <tr>
      <th>Soccer Field</th>
      <td>7</td>
    </tr>
    <tr>
      <th>Ethiopian Restaurant</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Tea Room</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Peruvian Restaurant</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Fried Chicken Joint</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Cuban Restaurant</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Indie Movie Theater</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Boxing Gym</th>
      <td>6</td>
    </tr>
    <tr>
      <th>BBQ Joint</th>
      <td>6</td>
    </tr>
    <tr>
      <th>History Museum</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Boat or Ferry</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Shoe Store</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Liquor Store</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Movie Theater</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Food Court</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Australian Restaurant</th>
      <td>5</td>
    </tr>
    <tr>
      <th>School</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Residential Building (Apartment / Condo)</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Flower Shop</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Burrito Place</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Video Game Store</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Concert Hall</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Event Space</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Health &amp; Beauty Service</th>
      <td>5</td>
    </tr>
    <tr>
      <th>African Restaurant</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Discount Store</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Monument / Landmark</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Women's Store</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Climbing Gym</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Thrift / Vintage Store</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Baseball Field</th>
      <td>4</td>
    </tr>
    <tr>
      <th>State / Provincial Park</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Sports Bar</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Falafel Restaurant</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Athletics &amp; Sports</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Harbor / Marina</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Whisky Bar</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Baseball Stadium</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Fountain</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Brewery</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Pet Café</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Gay Bar</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Convenience Store</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Museum</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Pilates Studio</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Roof Deck</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Cupcake Shop</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Bubble Tea Shop</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Comfort Food Restaurant</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Basketball Court</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Accessories Store</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Brazilian Restaurant</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Music Venue</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Other Great Outdoors</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Sake Bar</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Supplement Shop</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Beer Garden</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Turkish Restaurant</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Paper / Office Supplies Store</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Office</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Sculpture Garden</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Martial Arts School</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Record Shop</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Toy / Game Store</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Szechuan Restaurant</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Snack Place</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Public Art</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Argentinian Restaurant</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Breakfast Spot</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Dumpling Restaurant</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Waterfront</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Building</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Beach</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Jewelry Store</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Poke Place</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Chocolate Shop</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Empanada Restaurant</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Smoke Shop</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Hotel Bar</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Animal Shelter</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Bridge</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Comic Shop</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Amphitheater</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Beer Store</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Resort</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Wings Joint</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Halal Restaurant</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Skating Rink</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Hawaiian Restaurant</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Flea Market</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Hot Dog Joint</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Miscellaneous Shop</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Rental Car Location</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Daycare</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Taiwanese Restaurant</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Kids Store</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Gymnastics Gym</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Recreation Center</th>
      <td>2</td>
    </tr>
    <tr>
      <th>South American Restaurant</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Karaoke Bar</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Scandinavian Restaurant</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Church</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Frozen Yogurt Shop</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Factory</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Dive Bar</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Garden Center</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Adult Boutique</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Shanghai Restaurant</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Moroccan Restaurant</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Israeli Restaurant</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Market</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Veterinarian</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Music Store</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Antique Shop</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Lighthouse</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Motorcycle Shop</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Pie Shop</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Austrian Restaurant</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Shipping Store</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Nail Salon</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Organic Grocery</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Volleyball Court</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Bike Trail</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Sporting Goods Shop</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Bike Rental / Bike Share</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Train Station</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Electronics Store</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Street Art</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Library</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Massage Studio</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Kitchen Supply Store</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Tiki Bar</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Candy Store</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Ukrainian Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Bike Shop</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Island</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Sports Club</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Health Food Store</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Yemeni Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Duty-free Shop</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Hardware Store</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Peruvian Roast Chicken Joint</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Tex-Mex Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Soup Place</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Hotpot Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Filipino Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>South Indian Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>German Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Winery</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Tour Provider</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Butcher</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Golf Driving Range</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Track Stadium</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Pedestrian Plaza</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Irish Pub</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Bistro</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Film Studio</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Coworking Space</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Newsstand</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Video Store</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Multiplex</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Mattress Store</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Puerto Rican Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Field</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Theme Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Rest Area</th>
      <td>1</td>
    </tr>
    <tr>
      <th>College Theater</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Creperie</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Drugstore</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Cantonese Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Rugby Pitch</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Science Museum</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Laundry Service</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Bowling Alley</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Fast Food Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Dim Sum Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Board Shop</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Lebanese Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>College Arts Building</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Doctor's Office</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Department Store</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Big Box Store</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Venezuelan Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Roller Rink</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Pet Service</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Soba Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Eastern European Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Heliport</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Kosher Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Theme Park Ride / Attraction</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Lingerie Store</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Entertainment Service</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Circus</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Jewish Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Noodle House</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Japanese Curry Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Shop &amp; Service</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Tennis Stadium</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Opera House</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Malay Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Molecular Gastronomy Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Russian Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Hobby Shop</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Non-Profit</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Swiss Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Udon Restaurant</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Fish Market</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



- Convert venues into variables (as columns) and create a new dataframe manhattan_onehot


```python
manhattan_onehot = pd.get_dummies(manhattan_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
manhattan_onehot['DistrictName'] = manhattan_venues['DistrictName'] 

# move neighborhood column to the first column
fixed_columns = [manhattan_onehot.columns[-1]] + list(manhattan_onehot.columns[:-1])
manhattan_onehot = manhattan_onehot[fixed_columns]

manhattan_onehot.shape
```




    (2805, 306)



- Group this dataframe by each district


```python
manhattan_grouped = manhattan_onehot.groupby('DistrictName').sum().reset_index()
manhattan_grouped.head(8)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DistrictName</th>
      <th>Accessories Store</th>
      <th>Adult Boutique</th>
      <th>African Restaurant</th>
      <th>American Restaurant</th>
      <th>Amphitheater</th>
      <th>Animal Shelter</th>
      <th>Antique Shop</th>
      <th>Argentinian Restaurant</th>
      <th>Art Gallery</th>
      <th>Art Museum</th>
      <th>Arts &amp; Crafts Store</th>
      <th>Asian Restaurant</th>
      <th>Athletics &amp; Sports</th>
      <th>Australian Restaurant</th>
      <th>Austrian Restaurant</th>
      <th>BBQ Joint</th>
      <th>Bagel Shop</th>
      <th>Bakery</th>
      <th>Bank</th>
      <th>Bar</th>
      <th>Baseball Field</th>
      <th>Baseball Stadium</th>
      <th>Basketball Court</th>
      <th>Beach</th>
      <th>Beer Bar</th>
      <th>Beer Garden</th>
      <th>Beer Store</th>
      <th>Big Box Store</th>
      <th>Bike Rental / Bike Share</th>
      <th>Bike Shop</th>
      <th>Bike Trail</th>
      <th>Bistro</th>
      <th>Board Shop</th>
      <th>Boat or Ferry</th>
      <th>Bookstore</th>
      <th>Boutique</th>
      <th>Bowling Alley</th>
      <th>Boxing Gym</th>
      <th>Brazilian Restaurant</th>
      <th>Breakfast Spot</th>
      <th>Brewery</th>
      <th>Bridge</th>
      <th>Bubble Tea Shop</th>
      <th>Building</th>
      <th>Burger Joint</th>
      <th>Burrito Place</th>
      <th>Butcher</th>
      <th>Café</th>
      <th>Candy Store</th>
      <th>Cantonese Restaurant</th>
      <th>Caribbean Restaurant</th>
      <th>Cheese Shop</th>
      <th>Chinese Restaurant</th>
      <th>Chocolate Shop</th>
      <th>Church</th>
      <th>Circus</th>
      <th>Climbing Gym</th>
      <th>Clothing Store</th>
      <th>Cocktail Bar</th>
      <th>Coffee Shop</th>
      <th>College Arts Building</th>
      <th>College Theater</th>
      <th>Comfort Food Restaurant</th>
      <th>Comic Shop</th>
      <th>Concert Hall</th>
      <th>Convenience Store</th>
      <th>Cosmetics Shop</th>
      <th>Coworking Space</th>
      <th>Creperie</th>
      <th>Cuban Restaurant</th>
      <th>Cupcake Shop</th>
      <th>Cycle Studio</th>
      <th>Dance Studio</th>
      <th>Daycare</th>
      <th>Deli / Bodega</th>
      <th>Department Store</th>
      <th>Dessert Shop</th>
      <th>Dim Sum Restaurant</th>
      <th>Diner</th>
      <th>Discount Store</th>
      <th>Dive Bar</th>
      <th>Doctor's Office</th>
      <th>Dog Run</th>
      <th>Donut Shop</th>
      <th>Drugstore</th>
      <th>Dumpling Restaurant</th>
      <th>Duty-free Shop</th>
      <th>Eastern European Restaurant</th>
      <th>Electronics Store</th>
      <th>Empanada Restaurant</th>
      <th>Entertainment Service</th>
      <th>Ethiopian Restaurant</th>
      <th>Event Space</th>
      <th>Exhibit</th>
      <th>Factory</th>
      <th>Falafel Restaurant</th>
      <th>Farmers Market</th>
      <th>Fast Food Restaurant</th>
      <th>Field</th>
      <th>Filipino Restaurant</th>
      <th>Film Studio</th>
      <th>Fish Market</th>
      <th>Flea Market</th>
      <th>Flower Shop</th>
      <th>Food &amp; Drink Shop</th>
      <th>Food Court</th>
      <th>Food Truck</th>
      <th>Fountain</th>
      <th>French Restaurant</th>
      <th>Fried Chicken Joint</th>
      <th>Frozen Yogurt Shop</th>
      <th>Furniture / Home Store</th>
      <th>Garden</th>
      <th>Garden Center</th>
      <th>Gastropub</th>
      <th>Gay Bar</th>
      <th>German Restaurant</th>
      <th>Gift Shop</th>
      <th>Golf Driving Range</th>
      <th>Gourmet Shop</th>
      <th>Greek Restaurant</th>
      <th>Grocery Store</th>
      <th>Gym</th>
      <th>Gym / Fitness Center</th>
      <th>Gymnastics Gym</th>
      <th>Halal Restaurant</th>
      <th>Harbor / Marina</th>
      <th>Hardware Store</th>
      <th>Hawaiian Restaurant</th>
      <th>Health &amp; Beauty Service</th>
      <th>Health Food Store</th>
      <th>Heliport</th>
      <th>Historic Site</th>
      <th>History Museum</th>
      <th>Hobby Shop</th>
      <th>Hot Dog Joint</th>
      <th>Hotel</th>
      <th>Hotel Bar</th>
      <th>Hotpot Restaurant</th>
      <th>Ice Cream Shop</th>
      <th>Indian Restaurant</th>
      <th>Indie Movie Theater</th>
      <th>Indie Theater</th>
      <th>Irish Pub</th>
      <th>Island</th>
      <th>Israeli Restaurant</th>
      <th>Italian Restaurant</th>
      <th>Japanese Curry Restaurant</th>
      <th>Japanese Restaurant</th>
      <th>Jazz Club</th>
      <th>Jewelry Store</th>
      <th>Jewish Restaurant</th>
      <th>Juice Bar</th>
      <th>Karaoke Bar</th>
      <th>Kids Store</th>
      <th>Kitchen Supply Store</th>
      <th>Korean Restaurant</th>
      <th>Kosher Restaurant</th>
      <th>Latin American Restaurant</th>
      <th>Laundry Service</th>
      <th>Lebanese Restaurant</th>
      <th>Library</th>
      <th>Lighthouse</th>
      <th>Lingerie Store</th>
      <th>Liquor Store</th>
      <th>Lounge</th>
      <th>Malay Restaurant</th>
      <th>Market</th>
      <th>Martial Arts School</th>
      <th>Massage Studio</th>
      <th>Mattress Store</th>
      <th>Mediterranean Restaurant</th>
      <th>Men's Store</th>
      <th>Mexican Restaurant</th>
      <th>Middle Eastern Restaurant</th>
      <th>Miscellaneous Shop</th>
      <th>Mobile Phone Shop</th>
      <th>Molecular Gastronomy Restaurant</th>
      <th>Monument / Landmark</th>
      <th>Moroccan Restaurant</th>
      <th>Motorcycle Shop</th>
      <th>Movie Theater</th>
      <th>Multiplex</th>
      <th>Museum</th>
      <th>Music Store</th>
      <th>Music Venue</th>
      <th>Nail Salon</th>
      <th>New American Restaurant</th>
      <th>Newsstand</th>
      <th>Non-Profit</th>
      <th>Noodle House</th>
      <th>Office</th>
      <th>Opera House</th>
      <th>Optical Shop</th>
      <th>Organic Grocery</th>
      <th>Other Great Outdoors</th>
      <th>Outdoor Sculpture</th>
      <th>Paper / Office Supplies Store</th>
      <th>Park</th>
      <th>Pedestrian Plaza</th>
      <th>Performing Arts Venue</th>
      <th>Peruvian Restaurant</th>
      <th>Peruvian Roast Chicken Joint</th>
      <th>Pet Café</th>
      <th>Pet Service</th>
      <th>Pet Store</th>
      <th>Pharmacy</th>
      <th>Pie Shop</th>
      <th>Pier</th>
      <th>Pilates Studio</th>
      <th>Pizza Place</th>
      <th>Playground</th>
      <th>Plaza</th>
      <th>Poke Place</th>
      <th>Pool</th>
      <th>Pub</th>
      <th>Public Art</th>
      <th>Puerto Rican Restaurant</th>
      <th>Ramen Restaurant</th>
      <th>Record Shop</th>
      <th>Recreation Center</th>
      <th>Rental Car Location</th>
      <th>Residential Building (Apartment / Condo)</th>
      <th>Resort</th>
      <th>Rest Area</th>
      <th>Restaurant</th>
      <th>Roller Rink</th>
      <th>Roof Deck</th>
      <th>Rugby Pitch</th>
      <th>Russian Restaurant</th>
      <th>Sake Bar</th>
      <th>Salad Place</th>
      <th>Salon / Barbershop</th>
      <th>Sandwich Place</th>
      <th>Scandinavian Restaurant</th>
      <th>Scenic Lookout</th>
      <th>School</th>
      <th>Science Museum</th>
      <th>Sculpture Garden</th>
      <th>Seafood Restaurant</th>
      <th>Shanghai Restaurant</th>
      <th>Shipping Store</th>
      <th>Shoe Store</th>
      <th>Shop &amp; Service</th>
      <th>Skating Rink</th>
      <th>Smoke Shop</th>
      <th>Snack Place</th>
      <th>Soba Restaurant</th>
      <th>Soccer Field</th>
      <th>Soup Place</th>
      <th>South American Restaurant</th>
      <th>South Indian Restaurant</th>
      <th>Southern / Soul Food Restaurant</th>
      <th>Spa</th>
      <th>Spanish Restaurant</th>
      <th>Speakeasy</th>
      <th>Sporting Goods Shop</th>
      <th>Sports Bar</th>
      <th>Sports Club</th>
      <th>State / Provincial Park</th>
      <th>Steakhouse</th>
      <th>Street Art</th>
      <th>Supermarket</th>
      <th>Supplement Shop</th>
      <th>Sushi Restaurant</th>
      <th>Swiss Restaurant</th>
      <th>Szechuan Restaurant</th>
      <th>Taco Place</th>
      <th>Taiwanese Restaurant</th>
      <th>Tapas Restaurant</th>
      <th>Tea Room</th>
      <th>Tennis Court</th>
      <th>Tennis Stadium</th>
      <th>Tex-Mex Restaurant</th>
      <th>Thai Restaurant</th>
      <th>Theater</th>
      <th>Theme Park Ride / Attraction</th>
      <th>Theme Restaurant</th>
      <th>Thrift / Vintage Store</th>
      <th>Tiki Bar</th>
      <th>Tour Provider</th>
      <th>Toy / Game Store</th>
      <th>Track</th>
      <th>Track Stadium</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Turkish Restaurant</th>
      <th>Udon Restaurant</th>
      <th>Ukrainian Restaurant</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Venezuelan Restaurant</th>
      <th>Veterinarian</th>
      <th>Video Game Store</th>
      <th>Video Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Volleyball Court</th>
      <th>Waterfront</th>
      <th>Whisky Bar</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Winery</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
      <th>Yemeni Restaurant</th>
      <th>Yoga Studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Battery Park City-Lower Manhattan</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Central Harlem North-Polo Grounds</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Central Harlem South</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Chinatown</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Clinton</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>East Harlem North</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>East Harlem South</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>East Village</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



- As we just need restaurants, we restrict the number of columns by retaining only venue category that is "Restaurant"


```python
mann = pd.DataFrame(manhattan_grouped, columns = ["DistrictName"]+ list(manhattan_grouped.columns[manhattan_grouped.columns.str.contains("Restaurant")]))
```


```python
mann["Number_of_restaurants"] = mann.sum(axis=1)
```


```python
extr_mann = pd.DataFrame(mann, columns = ["DistrictName", "Number_of_restaurants"])
extr_mann.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DistrictName</th>
      <th>Number_of_restaurants</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Battery Park City-Lower Manhattan</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Central Harlem North-Polo Grounds</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Central Harlem South</td>
      <td>26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Chinatown</td>
      <td>29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Clinton</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_manh=pd.merge(manhattan_data,extr_mann, on='DistrictName')
```


```python
df_manh.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>DistrictCode</th>
      <th>DistrictName</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Population</th>
      <th>min_dist_to_subways_km</th>
      <th>number_subways_1.5km</th>
      <th>Number_of_restaurants</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Manhattan</td>
      <td>MN15</td>
      <td>Clinton</td>
      <td>40.766378</td>
      <td>-73.996548</td>
      <td>45884</td>
      <td>0.990834</td>
      <td>13</td>
      <td>27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Manhattan</td>
      <td>MN25</td>
      <td>Battery Park City-Lower Manhattan</td>
      <td>40.694739</td>
      <td>-74.001444</td>
      <td>39699</td>
      <td>0.767109</td>
      <td>13</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Manhattan</td>
      <td>MN14</td>
      <td>Lincoln Square</td>
      <td>40.775236</td>
      <td>-73.988364</td>
      <td>61489</td>
      <td>0.555412</td>
      <td>9</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Manhattan</td>
      <td>MN17</td>
      <td>Midtown-Midtown South</td>
      <td>40.756759</td>
      <td>-73.982858</td>
      <td>28630</td>
      <td>0.245619</td>
      <td>32</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Manhattan</td>
      <td>MN40</td>
      <td>Upper East Side-Carnegie Hill</td>
      <td>40.775283</td>
      <td>-73.960816</td>
      <td>61207</td>
      <td>0.201265</td>
      <td>10</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>



- Adding to the table the distance between each district and the nearest restaurant


```python
res, nb = [], []
for lat, lon, distr in zip(df_manh['Latitude'], df_manh['Longitude'], df_manh['DistrictName']):
    z = []
    k = 0
    for lat1, lon1, cat, dis in zip(manhattan_venues['Venue Latitude'], manhattan_venues['Venue Longitude'], manhattan_venues["Venue Category"], manhattan_venues["DistrictName"] ):
        if "Restaurant" in cat and distr==dis :
            dist = Distance(lat1, lon1, lat, lon)
            if dist<1: k = k+1
            z.append(dist)
    res.append(min(z))
    nb.append(k)
```


```python
df_manh["min_dist_to_restau_km"] = res
```

Now we generate our table which will be cluterised to determine the best places


```python
df_manh.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>DistrictCode</th>
      <th>DistrictName</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Population</th>
      <th>min_dist_to_subways_km</th>
      <th>number_subways_1.5km</th>
      <th>Number_of_restaurants</th>
      <th>min_dist_to_restau_km</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Manhattan</td>
      <td>MN15</td>
      <td>Clinton</td>
      <td>40.766378</td>
      <td>-73.996548</td>
      <td>45884</td>
      <td>0.990834</td>
      <td>13</td>
      <td>27</td>
      <td>0.448717</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Manhattan</td>
      <td>MN25</td>
      <td>Battery Park City-Lower Manhattan</td>
      <td>40.694739</td>
      <td>-74.001444</td>
      <td>39699</td>
      <td>0.767109</td>
      <td>13</td>
      <td>20</td>
      <td>0.156590</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Manhattan</td>
      <td>MN14</td>
      <td>Lincoln Square</td>
      <td>40.775236</td>
      <td>-73.988364</td>
      <td>61489</td>
      <td>0.555412</td>
      <td>9</td>
      <td>15</td>
      <td>0.477536</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Manhattan</td>
      <td>MN17</td>
      <td>Midtown-Midtown South</td>
      <td>40.756759</td>
      <td>-73.982858</td>
      <td>28630</td>
      <td>0.245619</td>
      <td>32</td>
      <td>17</td>
      <td>0.154862</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Manhattan</td>
      <td>MN40</td>
      <td>Upper East Side-Carnegie Hill</td>
      <td>40.775283</td>
      <td>-73.960816</td>
      <td>61207</td>
      <td>0.201265</td>
      <td>10</td>
      <td>23</td>
      <td>0.168686</td>
    </tr>
  </tbody>
</table>
</div>



# First, We use K-means Algorithm


```python
# set number of clusters
kclusters = 4

manhattan_clustering = df_manh.drop(['Borough',"DistrictCode","DistrictName"] , 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(manhattan_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
```




    array([3, 3, 2, 0, 2, 2, 3, 2, 2, 1])




```python
manhattan_clustering.insert(0, 'Cluster Labels', kmeans.labels_)
```


```python
manhattan_clustering["DistrictName"] = df_manh["DistrictName"]
```


```python
manhattan_clustering.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cluster Labels</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Population</th>
      <th>min_dist_to_subways_km</th>
      <th>number_subways_1.5km</th>
      <th>Number_of_restaurants</th>
      <th>min_dist_to_restau_km</th>
      <th>DistrictName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>40.766378</td>
      <td>-73.996548</td>
      <td>45884</td>
      <td>0.990834</td>
      <td>13</td>
      <td>27</td>
      <td>0.448717</td>
      <td>Clinton</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>40.694739</td>
      <td>-74.001444</td>
      <td>39699</td>
      <td>0.767109</td>
      <td>13</td>
      <td>20</td>
      <td>0.156590</td>
      <td>Battery Park City-Lower Manhattan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>40.775236</td>
      <td>-73.988364</td>
      <td>61489</td>
      <td>0.555412</td>
      <td>9</td>
      <td>15</td>
      <td>0.477536</td>
      <td>Lincoln Square</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>40.756759</td>
      <td>-73.982858</td>
      <td>28630</td>
      <td>0.245619</td>
      <td>32</td>
      <td>17</td>
      <td>0.154862</td>
      <td>Midtown-Midtown South</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>40.775283</td>
      <td>-73.960816</td>
      <td>61207</td>
      <td>0.201265</td>
      <td>10</td>
      <td>23</td>
      <td>0.168686</td>
      <td>Upper East Side-Carnegie Hill</td>
    </tr>
  </tbody>
</table>
</div>



# Results 

## Clusters Visualization with KMeans Algorithm


```python
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(manhattan_clustering['Latitude'], manhattan_clustering['Longitude'], manhattan_clustering['DistrictName'], manhattan_clustering['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=%3C%21DOCTYPE%20html%3E%0A%3Chead%3E%20%20%20%20%0A%20%20%20%20%3Cmeta%20http-equiv%3D%22content-type%22%20content%3D%22text/html%3B%20charset%3DUTF-8%22%20/%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%3Cscript%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20L_NO_TOUCH%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20L_DISABLE_3D%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%3C/script%3E%0A%20%20%20%20%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//code.jquery.com/jquery-1.12.4.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js%22%3E%3C/script%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//rawcdn.githack.com/python-visualization/folium/master/folium/templates/leaflet.awesome.rotate.css%22/%3E%0A%20%20%20%20%3Cstyle%3Ehtml%2C%20body%20%7Bwidth%3A%20100%25%3Bheight%3A%20100%25%3Bmargin%3A%200%3Bpadding%3A%200%3B%7D%3C/style%3E%0A%20%20%20%20%3Cstyle%3E%23map%20%7Bposition%3Aabsolute%3Btop%3A0%3Bbottom%3A0%3Bright%3A0%3Bleft%3A0%3B%7D%3C/style%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cmeta%20name%3D%22viewport%22%20content%3D%22width%3Ddevice-width%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20initial-scale%3D1.0%2C%20maximum-scale%3D1.0%2C%20user-scalable%3Dno%22%20/%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cstyle%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%23map_ae1aea8aaed045f1b92341651f7dac3b%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20position%3A%20relative%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20width%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20height%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20left%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20top%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%3C/style%3E%0A%20%20%20%20%20%20%20%20%0A%3C/head%3E%0A%3Cbody%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cdiv%20class%3D%22folium-map%22%20id%3D%22map_ae1aea8aaed045f1b92341651f7dac3b%22%20%3E%3C/div%3E%0A%20%20%20%20%20%20%20%20%0A%3C/body%3E%0A%3Cscript%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20map_ae1aea8aaed045f1b92341651f7dac3b%20%3D%20L.map%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22map_ae1aea8aaed045f1b92341651f7dac3b%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20center%3A%20%5B40.7896239%2C%20-73.9598939%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20crs%3A%20L.CRS.EPSG3857%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoom%3A%2011%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoomControl%3A%20true%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20preferCanvas%3A%20false%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20tile_layer_460a8fbaddbe4bfcac769918a9294df8%20%3D%20L.tileLayer%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22https%3A//%7Bs%7D.tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22attribution%22%3A%20%22Data%20by%20%5Cu0026copy%3B%20%5Cu003ca%20href%3D%5C%22http%3A//openstreetmap.org%5C%22%5Cu003eOpenStreetMap%5Cu003c/a%5Cu003e%2C%20under%20%5Cu003ca%20href%3D%5C%22http%3A//www.openstreetmap.org/copyright%5C%22%5Cu003eODbL%5Cu003c/a%5Cu003e.%22%2C%20%22detectRetina%22%3A%20false%2C%20%22maxNativeZoom%22%3A%2018%2C%20%22maxZoom%22%3A%2018%2C%20%22minZoom%22%3A%200%2C%20%22noWrap%22%3A%20false%2C%20%22opacity%22%3A%201%2C%20%22subdomains%22%3A%20%22abc%22%2C%20%22tms%22%3A%20false%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_31c1dfcb427d44818bb840fb1620e6df%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.7663775596076%2C%20-73.99654837674895%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23d4dd80%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23d4dd80%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4b98d28ef5104059b1e30ed1c4d4bd1d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0ee5004eadcd4318815f31756e1209dc%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_0ee5004eadcd4318815f31756e1209dc%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EClinton%20Cluster%203%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4b98d28ef5104059b1e30ed1c4d4bd1d.setContent%28html_0ee5004eadcd4318815f31756e1209dc%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_31c1dfcb427d44818bb840fb1620e6df.bindPopup%28popup_4b98d28ef5104059b1e30ed1c4d4bd1d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ab12c3354e89412f88584600d8173276%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.69473916696892%2C%20-74.00144418241418%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23d4dd80%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23d4dd80%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_00702d8a9d5e4888b1133479c2ef8664%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6dd7d965ef0b4f4e832eacf5a87994dc%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_6dd7d965ef0b4f4e832eacf5a87994dc%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBattery%20Park%20City-Lower%20Manhattan%20Cluster%203%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_00702d8a9d5e4888b1133479c2ef8664.setContent%28html_6dd7d965ef0b4f4e832eacf5a87994dc%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ab12c3354e89412f88584600d8173276.bindPopup%28popup_00702d8a9d5e4888b1133479c2ef8664%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e00e9f2833e649ff8452580a3ce1673e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.77523608214009%2C%20-73.98836366766845%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%232adddd%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%232adddd%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e6ab03f794c64999b2e287f112d975af%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_cdacd46b8b5145a493d5e48f0a004a44%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_cdacd46b8b5145a493d5e48f0a004a44%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELincoln%20Square%20Cluster%202%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e6ab03f794c64999b2e287f112d975af.setContent%28html_cdacd46b8b5145a493d5e48f0a004a44%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e00e9f2833e649ff8452580a3ce1673e.bindPopup%28popup_e6ab03f794c64999b2e287f112d975af%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_aa8a5f3c30014b6b8eec619621a0cc78%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.75675923700698%2C%20-73.98285820774203%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_0e8d34ca39634a39b324ae33c82648ac%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1d91c59271ba44df91416f8734745399%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1d91c59271ba44df91416f8734745399%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMidtown-Midtown%20South%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_0e8d34ca39634a39b324ae33c82648ac.setContent%28html_1d91c59271ba44df91416f8734745399%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_aa8a5f3c30014b6b8eec619621a0cc78.bindPopup%28popup_0e8d34ca39634a39b324ae33c82648ac%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_73ac759730474316bbf95a54fb140a98%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.77528339839104%2C%20-73.96081636591553%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%232adddd%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%232adddd%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3b52b8bd0acd4a98bcc346eb7d6eab5b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_498ee99c120e4176a221bcdfe5f2038a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_498ee99c120e4176a221bcdfe5f2038a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EUpper%20East%20Side-Carnegie%20Hill%20Cluster%202%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3b52b8bd0acd4a98bcc346eb7d6eab5b.setContent%28html_498ee99c120e4176a221bcdfe5f2038a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_73ac759730474316bbf95a54fb140a98.bindPopup%28popup_3b52b8bd0acd4a98bcc346eb7d6eab5b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5eaa99c9ec764a98a91fb79c9379e69b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.78050399554715%2C%20-73.93852233514457%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%232adddd%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%232adddd%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f77a77d6d93b4b04b6678dd39dc48d36%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c2020639878f4a32b1b6c68496595a0d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c2020639878f4a32b1b6c68496595a0d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EYorkville%20Cluster%202%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f77a77d6d93b4b04b6678dd39dc48d36.setContent%28html_c2020639878f4a32b1b6c68496595a0d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_5eaa99c9ec764a98a91fb79c9379e69b.bindPopup%28popup_f77a77d6d93b4b04b6678dd39dc48d36%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4dd1ddad25d34328a63069aa212a828e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.85601276666469%2C%20-73.92208086039953%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23d4dd80%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23d4dd80%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_901d77d525e441b0b96f4fc93f138ada%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ea7bdc85cf244661a873d74fa35c759f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ea7bdc85cf244661a873d74fa35c759f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMarble%20Hill-Inwood%20Cluster%203%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_901d77d525e441b0b96f4fc93f138ada.setContent%28html_ea7bdc85cf244661a873d74fa35c759f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4dd1ddad25d34328a63069aa212a828e.bindPopup%28popup_901d77d525e441b0b96f4fc93f138ada%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a600fb5951fa4c6bb351639fb7954c10%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.74911729284833%2C%20-74.00593919513614%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%232adddd%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%232adddd%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3eed99a174c24039aabf29bd10d9ba89%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_56719745c51a42e4bdc9108a01cc1646%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_56719745c51a42e4bdc9108a01cc1646%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHudson%20Yards-Chelsea-Flatiron-Union%20Square%20Cluster%202%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3eed99a174c24039aabf29bd10d9ba89.setContent%28html_56719745c51a42e4bdc9108a01cc1646%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a600fb5951fa4c6bb351639fb7954c10.bindPopup%28popup_3eed99a174c24039aabf29bd10d9ba89%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5b2d85aa8c75405bb6bd5a7307469264%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.734565184788664%2C%20-74.00621066209094%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%232adddd%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%232adddd%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b3b7294f6acd4801a4fd0f96f54f020e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9dcb4401391241d59029a8b63b03c128%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9dcb4401391241d59029a8b63b03c128%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWest%20Village%20Cluster%202%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b3b7294f6acd4801a4fd0f96f54f020e.setContent%28html_9dcb4401391241d59029a8b63b03c128%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_5b2d85aa8c75405bb6bd5a7307469264.bindPopup%28popup_b3b7294f6acd4801a4fd0f96f54f020e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_42da093d295345c3bfaa6143df86099f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.79083355054379%2C%20-73.97692336107843%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%238000ff%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%238000ff%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_13848a1d969448dea5c2adaf3a79ffd5%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7515791ec679408f9bb3fced8afc91af%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7515791ec679408f9bb3fced8afc91af%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EUpper%20West%20Side%20Cluster%201%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_13848a1d969448dea5c2adaf3a79ffd5.setContent%28html_7515791ec679408f9bb3fced8afc91af%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_42da093d295345c3bfaa6143df86099f.bindPopup%28popup_13848a1d969448dea5c2adaf3a79ffd5%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_53070a727b624d35b81865fbcb5bda18%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.827993805267916%2C%20-73.95054568805168%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23d4dd80%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23d4dd80%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_cbf02204fd584c199795d28d02404f4a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_45cb0894200344e892ffa9379def3bb7%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_45cb0894200344e892ffa9379def3bb7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHamilton%20Heights%20Cluster%203%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_cbf02204fd584c199795d28d02404f4a.setContent%28html_45cb0894200344e892ffa9379def3bb7%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_53070a727b624d35b81865fbcb5bda18.bindPopup%28popup_cbf02204fd584c199795d28d02404f4a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c2e568aa16984ad3863712f5a448f1c4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.78819344298152%2C%20-73.9423712473909%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23d4dd80%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23d4dd80%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_90841dc104a44644bc45c77e0bb05acd%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_65689ec3217144498a292d3303cc4f11%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_65689ec3217144498a292d3303cc4f11%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEast%20Harlem%20South%20Cluster%203%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_90841dc104a44644bc45c77e0bb05acd.setContent%28html_65689ec3217144498a292d3303cc4f11%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c2e568aa16984ad3863712f5a448f1c4.bindPopup%28popup_90841dc104a44644bc45c77e0bb05acd%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fe6bbe9d849b41febd812b6504151ef8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.68548712038733%2C%20-74.0095551305848%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_31f53157f188469c936d8297ac50a879%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ba0ae0c9a25d462fab21255c6cb89da3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ba0ae0c9a25d462fab21255c6cb89da3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3Epark-cemetery-etc-Manhattan%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_31f53157f188469c936d8297ac50a879.setContent%28html_ba0ae0c9a25d462fab21255c6cb89da3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_fe6bbe9d849b41febd812b6504151ef8.bindPopup%28popup_31f53157f188469c936d8297ac50a879%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b9b484bf85a14bd68ca84e153595419c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.80347835059119%2C%20-73.95096528365056%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23d4dd80%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23d4dd80%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d4dc07af84d5473ba122010a96ca314b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d95192aa11344f7c8bff19dd65d3af3e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d95192aa11344f7c8bff19dd65d3af3e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECentral%20Harlem%20South%20Cluster%203%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d4dc07af84d5473ba122010a96ca314b.setContent%28html_d95192aa11344f7c8bff19dd65d3af3e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b9b484bf85a14bd68ca84e153595419c.bindPopup%28popup_d4dc07af84d5473ba122010a96ca314b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_40fa0e8311774dacbc9d65185562c76f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.80515483757544%2C%20-73.93494660371782%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23d4dd80%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23d4dd80%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b80ba0c0d7f5489c9087ce9bbfe07d97%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_bbf18b502df54848ad5cab847578a192%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_bbf18b502df54848ad5cab847578a192%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEast%20Harlem%20North%20Cluster%203%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b80ba0c0d7f5489c9087ce9bbfe07d97.setContent%28html_bbf18b502df54848ad5cab847578a192%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_40fa0e8311774dacbc9d65185562c76f.bindPopup%28popup_b80ba0c0d7f5489c9087ce9bbfe07d97%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b86440f4cc08482f8270d13c889c4752%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.72734350120654%2C%20-73.98635207212405%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23d4dd80%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23d4dd80%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_28448f6861ac4f2d95f02da3d813fe72%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_adf7044468e64df894255d28d98ee02a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_adf7044468e64df894255d28d98ee02a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEast%20Village%20Cluster%203%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_28448f6861ac4f2d95f02da3d813fe72.setContent%28html_adf7044468e64df894255d28d98ee02a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b86440f4cc08482f8270d13c889c4752.bindPopup%28popup_28448f6861ac4f2d95f02da3d813fe72%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e5b9d801011b472aae1408980dcdfb03%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.705687831163246%2C%20-73.98203911644805%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%232adddd%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%232adddd%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e1a4430329934063892847c6de28bbb4%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5df13d5805894359b425f88a9d7f36f4%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_5df13d5805894359b425f88a9d7f36f4%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELower%20East%20Side%20Cluster%202%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e1a4430329934063892847c6de28bbb4.setContent%28html_5df13d5805894359b425f88a9d7f36f4%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e5b9d801011b472aae1408980dcdfb03.bindPopup%28popup_e1a4430329934063892847c6de28bbb4%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_107105bdc6bf4dab8d6201dbd38a2d0e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.73982212741276%2C%20-73.97474305311528%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23d4dd80%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23d4dd80%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_84b63afc829e42e1bd5ff0e1cb0761bc%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_fb2c6693b3144cf3bb0f9800238a73be%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_fb2c6693b3144cf3bb0f9800238a73be%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMurray%20Hill-Kips%20Bay%20Cluster%203%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_84b63afc829e42e1bd5ff0e1cb0761bc.setContent%28html_fb2c6693b3144cf3bb0f9800238a73be%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_107105bdc6bf4dab8d6201dbd38a2d0e.bindPopup%28popup_84b63afc829e42e1bd5ff0e1cb0761bc%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_de7c1e7e1ab44dc29b44ff3ba2c9fa16%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.808454248987175%2C%20-73.96406710398044%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23d4dd80%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23d4dd80%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_250bac394756495c8cf177bec3efd557%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_53ccec3fe3e94e9197eb51e6fd0d9277%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_53ccec3fe3e94e9197eb51e6fd0d9277%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMorningside%20Heights%20Cluster%203%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_250bac394756495c8cf177bec3efd557.setContent%28html_53ccec3fe3e94e9197eb51e6fd0d9277%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_de7c1e7e1ab44dc29b44ff3ba2c9fa16.bindPopup%28popup_250bac394756495c8cf177bec3efd557%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0eb4189e1d114e79ae80e9d35dd0534c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.85765544136461%2C%20-73.93734644214263%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%232adddd%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%232adddd%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_da17e60fe0c5462caa8766e40ca7eb81%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_479f5e245b0444e5b7c344e3fc1919d2%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_479f5e245b0444e5b7c344e3fc1919d2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWashington%20Heights%20North%20Cluster%202%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_da17e60fe0c5462caa8766e40ca7eb81.setContent%28html_479f5e245b0444e5b7c344e3fc1919d2%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_0eb4189e1d114e79ae80e9d35dd0534c.bindPopup%28popup_da17e60fe0c5462caa8766e40ca7eb81%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c2cc6e48df154426a9b112ca1431aa3c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.84274234313284%2C%20-73.9433111501022%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%232adddd%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%232adddd%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6ee0cba6fe364e33937b3aaae3d77ccf%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_acd3a0b7f525401c9557a2a0f2251bb1%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_acd3a0b7f525401c9557a2a0f2251bb1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWashington%20Heights%20South%20Cluster%202%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6ee0cba6fe364e33937b3aaae3d77ccf.setContent%28html_acd3a0b7f525401c9557a2a0f2251bb1%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c2cc6e48df154426a9b112ca1431aa3c.bindPopup%28popup_6ee0cba6fe364e33937b3aaae3d77ccf%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4ccb39c40cd74edd80d2cb5e3d077646%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.74880110296486%2C%20-73.96949297606015%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23d4dd80%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23d4dd80%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_dccb3328ca134c08ae58cca942204d27%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b664208a520242e7b954852710f28eaf%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b664208a520242e7b954852710f28eaf%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ETurtle%20Bay-East%20Midtown%20Cluster%203%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_dccb3328ca134c08ae58cca942204d27.setContent%28html_b664208a520242e7b954852710f28eaf%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4ccb39c40cd74edd80d2cb5e3d077646.bindPopup%28popup_dccb3328ca134c08ae58cca942204d27%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5e0ab7be33684360b2427aa35d6db6d5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.748395140173336%2C%20-73.95764383283307%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%232adddd%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%232adddd%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_aa64dc028d44427dbfc685a57232fd7a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e3ee7a17098248038b436f4231040c3f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e3ee7a17098248038b436f4231040c3f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELenox%20Hill-Roosevelt%20Island%20Cluster%202%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_aa64dc028d44427dbfc685a57232fd7a.setContent%28html_e3ee7a17098248038b436f4231040c3f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_5e0ab7be33684360b2427aa35d6db6d5.bindPopup%28popup_aa64dc028d44427dbfc685a57232fd7a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_61746d1e8d2046de9df84ef715d2ac4c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.72044868765068%2C%20-74.0071853926249%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23d4dd80%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23d4dd80%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4dc651d0dfa9446296ff2f01b9da1c5a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9538e33e5a2c44c5910ccf34d61ba610%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9538e33e5a2c44c5910ccf34d61ba610%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESoHo-TriBeCa-Civic%20Center-Little%20Italy%20Cluster%203%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4dc651d0dfa9446296ff2f01b9da1c5a.setContent%28html_9538e33e5a2c44c5910ccf34d61ba610%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_61746d1e8d2046de9df84ef715d2ac4c.bindPopup%28popup_4dc651d0dfa9446296ff2f01b9da1c5a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b13b448ce62042bfa7a56f6ac1daadbf%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.7138072360246%2C%20-73.99623982171583%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23d4dd80%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23d4dd80%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a1f070542ee34b9184b4d4cc35698341%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0aac34bd28474716947198516831e6a7%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_0aac34bd28474716947198516831e6a7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EChinatown%20Cluster%203%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a1f070542ee34b9184b4d4cc35698341.setContent%28html_0aac34bd28474716947198516831e6a7%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b13b448ce62042bfa7a56f6ac1daadbf.bindPopup%28popup_a1f070542ee34b9184b4d4cc35698341%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_89cd322442b447fb808b997dcc91da85%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.82607099769937%2C%20-73.93699150049123%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%232adddd%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%232adddd%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_338651559914406b82147f16295dd64e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_24f886fb6156485a809e7dd6278842a5%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_24f886fb6156485a809e7dd6278842a5%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECentral%20Harlem%20North-Polo%20Grounds%20Cluster%202%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_338651559914406b82147f16295dd64e.setContent%28html_24f886fb6156485a809e7dd6278842a5%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_89cd322442b447fb808b997dcc91da85.bindPopup%28popup_338651559914406b82147f16295dd64e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0d9c1f4fa11b440bb7c45138878517ec%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.819604017014434%2C%20-73.9561130807665%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_0ed916a152a343b5aff5ae1b8534af07%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e34c6ece903347559d95221d107e66b9%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e34c6ece903347559d95221d107e66b9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EManhattanville%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_0ed916a152a343b5aff5ae1b8534af07.setContent%28html_e34c6ece903347559d95221d107e66b9%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_0d9c1f4fa11b440bb7c45138878517ec.bindPopup%28popup_0ed916a152a343b5aff5ae1b8534af07%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_080d12302a85455d952111f7a887e03c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.736507018580056%2C%20-73.98381323573406%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ac000ad3d7a34377b474b209d2b701ab%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d78fa2ec749c4e3c9a0707a023cdd579%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d78fa2ec749c4e3c9a0707a023cdd579%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGramercy%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ac000ad3d7a34377b474b209d2b701ab.setContent%28html_d78fa2ec749c4e3c9a0707a023cdd579%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_080d12302a85455d952111f7a887e03c.bindPopup%28popup_ac000ad3d7a34377b474b209d2b701ab%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f85302962c7d42da974b9bc748066e1c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B40.73168015130651%2C%20-73.97437379671852%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_ae1aea8aaed045f1b92341651f7dac3b%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_7818ca359f894eb0ac809125fc239e0d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_867eb229e55b4281b229cc332e50455e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_867eb229e55b4281b229cc332e50455e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EStuyvesant%20Town-Cooper%20Village%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_7818ca359f894eb0ac809125fc239e0d.setContent%28html_867eb229e55b4281b229cc332e50455e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f85302962c7d42da974b9bc748066e1c.bindPopup%28popup_7818ca359f894eb0ac809125fc239e0d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%3C/script%3E onload="this.contentDocument.open();this.contentDocument.write(    decodeURIComponent(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



- Clusters 0


```python
manhattan_clustering.loc[manhattan_clustering['Cluster Labels'] == 0, manhattan_clustering.columns[[manhattan_clustering.shape[1]-1] + list(range(3, manhattan_clustering.shape[1]-1))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DistrictName</th>
      <th>Population</th>
      <th>min_dist_to_subways_km</th>
      <th>number_subways_1.5km</th>
      <th>Number_of_restaurants</th>
      <th>min_dist_to_restau_km</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Midtown-Midtown South</td>
      <td>28630</td>
      <td>0.245619</td>
      <td>32</td>
      <td>17</td>
      <td>0.154862</td>
    </tr>
    <tr>
      <th>12</th>
      <td>park-cemetery-etc-Manhattan</td>
      <td>1849</td>
      <td>1.361403</td>
      <td>1</td>
      <td>13</td>
      <td>0.584997</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Manhattanville</td>
      <td>22950</td>
      <td>0.336913</td>
      <td>8</td>
      <td>31</td>
      <td>0.124191</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Gramercy</td>
      <td>27988</td>
      <td>0.440937</td>
      <td>22</td>
      <td>36</td>
      <td>0.105912</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Stuyvesant Town-Cooper Village</td>
      <td>21049</td>
      <td>0.620675</td>
      <td>7</td>
      <td>26</td>
      <td>0.569645</td>
    </tr>
  </tbody>
</table>
</div>



- Clusters 1


```python
manhattan_clustering.loc[manhattan_clustering['Cluster Labels'] == 1, manhattan_clustering.columns[[manhattan_clustering.shape[1]-1] + list(range(3, manhattan_clustering.shape[1]-1))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DistrictName</th>
      <th>Population</th>
      <th>min_dist_to_subways_km</th>
      <th>number_subways_1.5km</th>
      <th>Number_of_restaurants</th>
      <th>min_dist_to_restau_km</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>Upper West Side</td>
      <td>132378</td>
      <td>0.250604</td>
      <td>9</td>
      <td>30</td>
      <td>0.248552</td>
    </tr>
  </tbody>
</table>
</div>



- Clusters 2


```python
manhattan_clustering.loc[manhattan_clustering['Cluster Labels'] == 2, manhattan_clustering.columns[[manhattan_clustering.shape[1]-1] + list(range(3, manhattan_clustering.shape[1]-1))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DistrictName</th>
      <th>Population</th>
      <th>min_dist_to_subways_km</th>
      <th>number_subways_1.5km</th>
      <th>Number_of_restaurants</th>
      <th>min_dist_to_restau_km</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Lincoln Square</td>
      <td>61489</td>
      <td>0.555412</td>
      <td>9</td>
      <td>15</td>
      <td>0.477536</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Upper East Side-Carnegie Hill</td>
      <td>61207</td>
      <td>0.201265</td>
      <td>10</td>
      <td>23</td>
      <td>0.168686</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Yorkville</td>
      <td>77942</td>
      <td>0.830499</td>
      <td>5</td>
      <td>24</td>
      <td>0.599430</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Hudson Yards-Chelsea-Flatiron-Union Square</td>
      <td>70150</td>
      <td>0.755112</td>
      <td>13</td>
      <td>11</td>
      <td>0.215781</td>
    </tr>
    <tr>
      <th>8</th>
      <td>West Village</td>
      <td>66880</td>
      <td>0.306087</td>
      <td>23</td>
      <td>35</td>
      <td>0.041516</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Lower East Side</td>
      <td>72957</td>
      <td>0.777100</td>
      <td>5</td>
      <td>7</td>
      <td>0.340188</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Washington Heights North</td>
      <td>67136</td>
      <td>0.306602</td>
      <td>7</td>
      <td>22</td>
      <td>0.264096</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Washington Heights South</td>
      <td>84438</td>
      <td>0.361357</td>
      <td>9</td>
      <td>29</td>
      <td>0.080978</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Lenox Hill-Roosevelt Island</td>
      <td>80771</td>
      <td>0.727095</td>
      <td>10</td>
      <td>28</td>
      <td>0.075408</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Central Harlem North-Polo Grounds</td>
      <td>75282</td>
      <td>0.247548</td>
      <td>15</td>
      <td>22</td>
      <td>0.247593</td>
    </tr>
  </tbody>
</table>
</div>



- Clusters 3


```python
manhattan_clustering.loc[manhattan_clustering['Cluster Labels'] == 3, manhattan_clustering.columns[[manhattan_clustering.shape[1]-1] + list(range(3, manhattan_clustering.shape[1]-1))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DistrictName</th>
      <th>Population</th>
      <th>min_dist_to_subways_km</th>
      <th>number_subways_1.5km</th>
      <th>Number_of_restaurants</th>
      <th>min_dist_to_restau_km</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Clinton</td>
      <td>45884</td>
      <td>0.990834</td>
      <td>13</td>
      <td>27</td>
      <td>0.448717</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Battery Park City-Lower Manhattan</td>
      <td>39699</td>
      <td>0.767109</td>
      <td>13</td>
      <td>20</td>
      <td>0.156590</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Marble Hill-Inwood</td>
      <td>46746</td>
      <td>0.580384</td>
      <td>11</td>
      <td>13</td>
      <td>0.578123</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Hamilton Heights</td>
      <td>48520</td>
      <td>0.161192</td>
      <td>10</td>
      <td>33</td>
      <td>0.081053</td>
    </tr>
    <tr>
      <th>11</th>
      <td>East Harlem South</td>
      <td>57902</td>
      <td>0.506402</td>
      <td>8</td>
      <td>24</td>
      <td>0.164852</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Central Harlem South</td>
      <td>43383</td>
      <td>0.190487</td>
      <td>15</td>
      <td>26</td>
      <td>0.105730</td>
    </tr>
    <tr>
      <th>14</th>
      <td>East Harlem North</td>
      <td>58019</td>
      <td>0.249872</td>
      <td>9</td>
      <td>28</td>
      <td>0.472816</td>
    </tr>
    <tr>
      <th>15</th>
      <td>East Village</td>
      <td>44136</td>
      <td>0.498885</td>
      <td>21</td>
      <td>36</td>
      <td>0.027682</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Murray Hill-Kips Bay</td>
      <td>50742</td>
      <td>0.879675</td>
      <td>13</td>
      <td>31</td>
      <td>0.193868</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Morningside Heights</td>
      <td>55929</td>
      <td>0.081502</td>
      <td>10</td>
      <td>33</td>
      <td>0.235167</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Turtle Bay-East Midtown</td>
      <td>51231</td>
      <td>0.624311</td>
      <td>11</td>
      <td>31</td>
      <td>0.137385</td>
    </tr>
    <tr>
      <th>23</th>
      <td>SoHo-TriBeCa-Civic Center-Little Italy</td>
      <td>42742</td>
      <td>0.128233</td>
      <td>34</td>
      <td>27</td>
      <td>0.130177</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Chinatown</td>
      <td>47844</td>
      <td>0.511021</td>
      <td>33</td>
      <td>29</td>
      <td>0.180366</td>
    </tr>
  </tbody>
</table>
</div>



# Discussion

As we can observe, the population of district have a high weight on the classification. The number of restaurants and subways are highly variable in each cluster. The cluster 1 has only one district and is the highly inhabited. Although it has 30 restaurants, the relationship between the population and the number of restaurant is good to allow installing an other restaurant. The cluster 2 contains the second category of well inhabited districts and some district such as Lower East Side has only 7 restaurants on a radius of 1 km. We cannot advise clustesr 0 and 3 which contain districts with few population and a lot of restaurants.

Our analysis did not take into account the repartition of population in a district. It based on the center of the Neighborhood Tabulation Areas. However, the results can be confirmed by the reality in Manhattan. As example, the cluster 3 which is not allowed, contains Manhattanville neighborhood which is few inhabited but has a lot of restaurants. It welcomes many tourists, likewise Gramercy.

### Future work

In terms of future work, I would be interested in applying the approach to other boroughs and also with other advanced machine learning techniques such as Density-based clustering (DBSCAN) to reinforce the clustering.



```python

```
