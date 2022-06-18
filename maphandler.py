# Python libs.:
import googlemaps
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import math
from random import random

# Local libs

# tile info.: https://www.maptiler.com/google-maps-coordinates-tile-bounds-projection/#3/15.00/50.00


class MapDownloader:
    def __init__(self, map_size=512):
        # variabless
        self._lat = None
        self._lng = None
        self._zoom = None

        self._map_size = map_size
        self._initialResolution = 2 * math.pi * 6378137 / self._map_size
        self._originShift = 2 * math.pi * 6378137 / 2.0
        self._API_KEY = ''
        self._URL = 'https://maps.googleapis.com/maps/api/staticmap?'

        self._height, self._width = 512, 512
        self._g_height, self._g_width = 256, 256

        # gmap instace
        self._gmap = googlemaps.Client(self._API_KEY)

        self._bounds = None

    def set_variables(self, lat, lon, zoom):
        self._lat = lat
        self._lng = lon
        self._zoom = zoom

    def get_map(self,map_type, img_name=False, to_rgb=True):

        src = \
            f'{self._URL}' \
            f'center={self._lat},{self._lng}' \
            f'&zoom={self._zoom}' \
            f'&size={self._map_size}x{self._map_size}' \
            f'&maptype={map_type}' \
            f'&key={self._API_KEY}' \
            f'&sensor=false'

        region_map_request = requests.get(src).content

        if img_name is not False:
            image_name = f'{img_name}'
            with open(f'data/maps/{image_name}.jpg', 'wb') as handler:
                handler.write(region_map_request)
            handler.close()

        region_map_request = requests.get(src).content

        if to_rgb:
            region_map =Image.open(BytesIO(region_map_request)).convert('RGB')

        else:
            region_map = Image.open(BytesIO(region_map_request))

        return region_map

    def latLngToPoint(self, lat, lng):

        x = (lng + 180) * (self._g_width / 360)
        y = ((1 - math.log(math.tan(lat * math.pi / 180) + 1 / math.cos(lat * math.pi / 180)) / math.pi) / 2) * self._g_height

        return x, y

    def pointToLatLng(self, x, y):
        lng = x / self._g_width * 360 - 180

        n = math.pi - 2 * math.pi * y / self._g_height
        lat = (180 / math.pi * math.atan(0.5 * (math.exp(n) - math.exp(-n))))

        return lat, lng

    def get_bounds(self):

        xScale = math.pow(2, self._zoom) / (self._map_size / self._g_height)
        yScale = math.pow(2, self._zoom) / (self._map_size / self._g_width)

        centreX, centreY = self.latLngToPoint(self._lat, self._lng)

        southWestX = centreX - (self._g_width / 2) / xScale
        southWestY = centreY + (self._g_height / 2) / yScale
        SWlat, SWlng = self.pointToLatLng(southWestX, southWestY)

        northEastX = centreX + (self._g_width / 2) / xScale
        northEastY = centreY - (self._g_height / 2) / yScale
        NElat, NElng = self.pointToLatLng(northEastX, northEastY)

        return [SWlat, NElat], [SWlng, NElng]
