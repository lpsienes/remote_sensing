# Actualizado 30-9-21
# # Librerías-------------------------------
import os 
import ee
import json
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from geog import propagate
import uuid
from IPython.display import display_html, display_javascript
from scipy.signal import savgol_filter
from ts_functions import *
plt.style.use('default')

ee.Initialize()

# FUNCIONES Y CLASES -------------------------------
# Las funciones siguientes calculan un estadístico sobre la selección de píxeles asignada a "geometry" en 
# la banda "band" con una resolución de "scale" metros. Como argumento "image" se le puede pasar tanto una
# única imagen como una colección de imágenes, caso en el cuál mapeará todas ellas. El resultado es un nuevo
# campo en el json de la imagen llamado: 'MEAN_B1' para el caso de cálculo de la media de la banda 1.

def meanBand(image, band, geometry, scale=30, maxPixels=1e10,):
    if type(geometry) == dict:
        geometry = ee.Geometry(geometry['geometry'])
    def foo(image):
        mean_band = image.select(band).reduceRegion(reducer = ee.Reducer.mean(),
                                   geometry = geometry,
                                   maxPixels = maxPixels,
                                   scale = scale).get(band)
        image = image.set('MEAN_' + band, mean_band)
        return image
    if type(image) ==  ee.imagecollection.ImageCollection: 
        return image.map(foo)
    else:
        return foo(image)

    
def medianBand(image, band, geometry, scale=30, maxPixels=1e10):
    def foo(image):
        median_band = image.select(band).reduceRegion(reducer = ee.Reducer.median(),
                                       geometry = geometry,
                                       maxPixels = maxPixels,
                                       scale = scale).get(band)
        image = image.set('MEDIAN_' + band, median_band)
        return image
    if type(image) ==  ee.imagecollection.ImageCollection: 
        return image.map(foo)
    else:
        return foo(image)
    
    
def stdevBand(image, band, geometry, scale=30, maxPixels=1e10):
    def foo (image):
        stdev_band = image.select(band).reduceRegion(reducer = ee.Reducer.stdDev(),
                                       geometry = geometry,
                                       maxPixels = maxPixels,
                                       scale = scale).get(band)
        image = image.set('STDEV_' + band, stdev_band)
        return image 
    if type(image) ==  ee.imagecollection.ImageCollection: 
        return image.map(foo)
    else:
        return foo(image)
    
    
def minBand(image, band, geometry, scale=30, maxPixels=1e10):
    def foo (image):
        min_band = image.select(band).reduceRegion(reducer = ee.Reducer.min(),
                                       geometry = geometry,
                                       maxPixels = maxPixels,
                                       scale = scale).get(band)
        image = image.set('MIN_' + band, min_band)
        return image 
    if type(image) ==  ee.imagecollection.ImageCollection: 
        return image.map(foo)
    else:
        return foo(image)   
    
    
def maxBand(image, band, geometry, scale=30, maxPixels=1e10):
    def foo (image):
        max_band = image.select(band).reduceRegion(reducer = ee.Reducer.max(),
                                       geometry = geometry,
                                       maxPixels = maxPixels,
                                       scale = scale).get(band)
        image = image.set('MAX_' + band, max_band)
        return image
    if type(image) ==  ee.imagecollection.ImageCollection: 
        return image.map(foo)
    else:
        return foo(image)  

def modeBand(image, band, geometry, scale=30, maxPixels=1e10):
    def foo (image):
        mode_band = image.select(band).reduceRegion(reducer = ee.Reducer.mode(),
                                       geometry = geometry,
                                       maxPixels = maxPixels,
                                       scale = scale).get(band)
        image = image.set('MODE_' + band, mode_band)
        return image
    if type(image) ==  ee.imagecollection.ImageCollection: 
        return image.map(foo)
    else:
        return foo(image)  

# Función que permite sacar por pantalla en formato JSON un diccionario
class RenderJSON(object):
    def __init__(self, json_data):
        if isinstance(json_data, dict):
            self.json_str = json.dumps(json_data)
        else:
            self.json_str = json
        self.uuid = str(uuid.uuid4())
        
    def _ipython_display_(self):
        display_html('<div id="{}" style="height: 600px; width:100%;"></div>'.format(self.uuid),
            raw=True
        )
        display_javascript("""
        require(["https://rawgit.com/caldwell/renderjson/master/renderjson.js"], function() {
          document.getElementById('%s').appendChild(renderjson(%s))
        });
        """ % (self.uuid, self.json_str), raw=True)
        
# La clase land estructura el objeto cultivo asignando un identificador, latitud, longitud y radio
class land:
    def __init__(self, Id, latitude, longitude, radius):
        self.id = Id
        self.longitude = longitude
        self.latitude = latitude
        self.radius = radius
        self.coordinates = [self.latitude, self.longitude]
        
# La función createCircle crea geometrías circulares al pasarle las coordenadas del centro y la medida del
#radio. El nº de puntos influirá en la resolución de la geometría
def createCircle(coordinates, radius, n_points=100):
    p = Point(coordinates)
    angles = np.linspace(0, n_points, n_points+1)
    polygon = propagate(p, angles, radius)
    polygon = ee.Geometry.Polygon(polygon.tolist())
    return polygon

# Índices de vegetación ---------------------------------------------------------------------------------------------------
def addNDVI(image, bands, *args):
    def foo(image):
        ndvi = image.expression('(nir-red)/(nir+red)',
            {'nir': image.select(bands['nir']), 'red': image.select(bands['red'])}).rename('NDVI')
        return image.addBands(ndvi)
    if type(image) ==  ee.imagecollection.ImageCollection: 
        return image.map(foo)
    else:
        return foo(image)

# Soil Adjusted Vegetation Index (SAVI) es similar al NDVI, pero el factor L amorgiua la presencia de suelo:
# desde 0 gran densidad vegetal hasta 1 para zonas con escasez.
# Por defecto L=0.5,cobertura foliar moderada
def addSAVI(image, bands, parameter, *args): 
    def foo(image):
        savi = image.expression('(1+L)*(nir-red)/(nir+red+L)',
            {'nir': image.select(bands['nir']), 'red': image.select(bands['red']),
             'L': parameter.get('SAVI_L', 0.5)}).rename('SAVI')
        return image.addBands(savi)
    if type(image) ==  ee.imagecollection.ImageCollection: 
        return image.map(foo)
    else:
        return foo(image)

# Modified Soil Adjusted Vegetation Index (MSAVI) reduce el efecto del suelo desnudo de SAVI
def addMSAVI(image, bands, parameter, *args):
    def foo(image):
        msavi = image.expression('0.5 * (2*nir + 1 - ((2*nir + 1)**2 - 8*(nir-red))**0.5)',
            {'nir': image.select(bands['nir']), 'red': image.select(bands['red'])}).rename('MSAVI')
        return image.addBands(msavi)
    if type(image) ==  ee.imagecollection.ImageCollection:
        return image.map(foo)
    else:
        return foo(image)

# Solo puede calcularse si satellite == sentinel (red1 = 700nm) Modified Chlorophyll Absorption Ratio Index 
def addMCARI(image, bands, *args):
    def foo(image):
        mcari = image.expression('((red1 - red) - 0.2 * (red1 - green)) * (red1 / red)',
            {'red1': image.select(bands['red1']), 'red': image.select(bands['red']), 'green': image.select(bands['green'])}).rename('MCARI')
        return image.addBands(mcari)
    if type(image) ==  ee.imagecollection.ImageCollection: 
        return image.map(foo)
    else:
        return foo(image)

# Otra definición de MCARI disponible también para los LANDSAT
def addMCARI2(image, bands, *args):
    def foo(image):
        mcari = image.expression('(1.5*(2.5*(nir-red)-1.3*(nir-green)))/((2*nir + 1)**2 - (6*nir-5*(red**0.5))-0.5)**0.5',
            {'nir': image.select(bands['nir']), 'red': image.select(bands['red']), 'green': image.select(bands['green'])}).rename('MCARI2')
        return image.addBands(mcari)
    if type(image) ==  ee.imagecollection.ImageCollection: 
        return image.map(foo)
    else:
        return foo(image)

# Atmospherically Resistant Vegetation Index  (ARVI)
def addARVI(image, bands, *args):
    def foo(image):
        arvi = image.expression('(nir-2*red + blue)/(nir + 2*red + blue)', {'nir': image.select(bands['nir']), 'red': image.select(bands['red']), 'blue': image.select(bands['blue'])}).rename('ARVI')
        return image.addBands(arvi)
    if type(image) ==  ee.imagecollection.ImageCollection: 
        return image.map(foo)
    else:
        return foo(image)

# Weighted Differenced Vegetation Index (WDVI) 
def addWDVI(image, bands, parameter, *args): # parameter pendiente de la línea de suelo
    def foo(image):
        if 'WDVI_slope' not in parameter.keys():
            print('WDVI slope not define: set to default 1')
        wdvi = image.expression('nir-m*red',
            {'nir': image.select(bands['nir']), 'red': image.select(bands['red']),
             'm': parameter.get('WDVI_slope', 1)}).rename('WDVI')
        return image.addBands(wdvi)
    if type(image) ==  ee.imagecollection.ImageCollection: 
        return image.map(foo)
    else:
        return foo(image)

#Transformed Normalized Difference Vegetation Index (TNDVI)
def addTNDVI(image, bands, *args):
    def foo(image):
        tndvi = image.expression('((nir-red)/(nir+red) + 0.5)**0.5',
            {'nir': image.select(bands['nir']), 'red': image.select(bands['red'])}).rename('TNDVI')
        return image.addBands(tndvi)
    if type(image) ==  ee.imagecollection.ImageCollection: 
        return image.map(foo)
    else:
        return foo(image)

# Green Normalized Difference Vegetation Index (GNDVI)
def addGNDVI(image, bands, *args):
    def foo(image):
        gndvi = image.expression('(nir-green)/(nir+green)',
            {'nir': image.select(bands['nir']), 'red': image.select(bands['red']),
             'green': image.select(bands['green'])}).rename('GNDVI')
        return image.addBands(gndvi)
    if type(image) ==  ee.imagecollection.ImageCollection: 
        return image.map(foo)
    else:
        return foo(image)
#-------------------------------------------------------------------------------------------------------------------------
# La función StatisticsBand devuelve la media, mediana, moda, min o max del valor de una banda en un área 
# determinada
# A las funciones las invocamos con stats: MEAN, MEDIAN, MIN, MAX, STDEV. El parámetro buffer_param define el
#área que queremos agregar (signo positivo) o quitar de la parcela (signo negativo) en tanto por 1. 
# Por ejemplor, si queremos reducir el borde en un 10% deberemos escribir buffer_param = -0.1:
def StatisticsBand(collection, band, feature, stats = 'MEAN', buffer_param = None):
    try: geometry = ee.Geometry(feature['geometry']);
    except: geometry = ee.Geometry(feature.getInfo()['geometry']);

    if buffer_param != None:
        try: buffer = feature['properties']['radius']*buffer_param
        except: buffer = feature.getInfo()['properties']['radius']*buffer_param
        geometry = geometry.buffer(buffer)
    inds = []; dates = []
    for i, image in enumerate(collection.getInfo()['features']):
        try: 
            inds.append(image['properties'][stats + '_' + band])
            dates.append(datetime.fromtimestamp(image['properties']['system:time_start']/1000))
        except:
            print('Error at dt=', datetime.fromtimestamp(image['properties']['system:time_start']/1000))
            pass
    return inds, dates
    
# FUNCIONES MONGO 
def getUniSeriesMongo(satellite, date_ini, date_fin, crops_json, band_name, stat):
    print('Existing bands:', ', '.join(bands.keys()), '\nSelecting:', band_name)
    lisJSON=[]
    for i, crop in enumerate(crops_json):
        print(i)
        func = statistics_function[stat]
        collection = ee.ImageCollection(satellite_dict[satellite]).filterDate(date_ini, date_fin)
        if band_name in vegetation_index.keys():
            band = band_name
            collection = vegetation_index[band](collection, bands, parameters)
        else: band = bands[band_name]

        collection = collection.select(band)
        feature_of_interest = ee.Feature(crops_json[i])
        collection_of_interest = func(collection.filterBounds(feature_of_interest.geometry()), band, feature_of_interest.geometry(), scale)
        band_values, dates = StatisticsBand(collection_of_interest, band, feature_of_interest, stat, -0.1)

        series=[{'date': dates[i].strftime("%Y-%m-%d"), band: band_values[i]} for i in range(len(dates))]

        label=crops_json[i]['properties']['id'] + '_' + satellite + '_' + band
        json={'_id':label,'field':crops_json[i]['properties']['id'],'satellite':satellite,
                'band':band,'series':series}
        lisJSON.append(json)
        print (json)
    #db[col].insert_many(lisJSON,ordered=False)


# Actualiza las series de las bandas e índices en MONGO
def updateMultiBandsMongo(satellite, date_ini, date_fin, crops_json, sel_bands, stat, mongo_collection, parameters={'WDVI_slope': 4, 'SAVI_L': 2}, scale=20):
    if datetime.strptime(date_fin, "%Y-%m-%d") < datetime.strptime(satellite.oldestDate, "%Y-%m-%d"):
        time_ini = satellite.oldestDate
        print(f'Initial date before: changing to {datetime.strptime(satellite.oldestDate, "%Y-%m-%d")}')

    time_fin = datetime.strptime(date_fin, "%Y-%m-%d")
    bands = satellite.bandsDict

    if  satellite.name=='landsat8':
        sel_bands.append('temperature')
        sel_bands.append('ultrablue')

    for i, crop in enumerate(crops_json):
        label=crops_json[i]['properties']['id'] + '_' + satellite.name + '_band'
        json={'_id':label,'field': crops_json[i]['properties']['id'],'satellite':satellite.name, 'type':'index'}
        print('\n', i, label)

        fields_to_update_set = []
        fields_to_update_push = []

        if mongo_collection.find_one({'_id': label}): #EL OBJETO YA EXISTE -> HARE UPDATE
            saved_dates = mongo_collection.find_one({'_id': label})['dates']
            len_dates = len(saved_dates)
            time_ini = saved_dates[-1] + timedelta(days=1)

            # Calculo bandas
            if time_ini.date() < time_fin.date():
                for band_name in sel_bands:
                    print(band_name)
                    if band_name in mongo_collection.find_one({'_id': label}).keys(): #UPDATE
                        len_band = len(mongo_collection.find_one({'_id': label})[band_name])
                        dif_len = len_dates - len_band
                        fields_to_update_push.append(band_name)
                        #print(len_band, len_dates)

                        if dif_len > 0: time_ini = saved_dates[-dif_len]
                        else: time_ini = saved_dates[-1] + timedelta(days=1)

                    else: #UPDATE SET
                        dif_len = len_dates
                        time_ini = saved_dates[0] 
                        fields_to_update_set.append(band_name)
                    
                    str_time_ini = time_ini.strftime("%Y-%m-%d")
                    str_time_fin = time_fin.strftime("%Y-%m-%d")

                    func = statistics_function[stat]
                    collection = ee.ImageCollection(satellite.collectionId).filterDate(str_time_ini, str_time_fin)
                    if band_name in vegetation_index.keys():
                        band = band_name
                        collection = vegetation_index[band](collection, bands, parameters)
                    else: band = bands[band_name]

                    collection = collection.select(band)
                    feature_of_interest = ee.Feature(crops_json[i])
                    collection_of_interest = func(collection.filterBounds(feature_of_interest.geometry()), band, feature_of_interest.geometry(), scale)
                    band_values, dates = StatisticsBand(collection_of_interest, band, feature_of_interest, stat, -0.1)

                    dates_to_update = [date for date in dates if date not in saved_dates]

                    bands_to_update = band_values
                    str_dates_to_update = [dates_to_update[i].strftime("%Y-%m-%d") for i in range(len(dates_to_update))]
                    json[band_name]=bands_to_update

                json['dates']=dates_to_update
                json['str_dates']=str_dates_to_update
                fields_to_update_push.append('dates')
                fields_to_update_push.append('str_dates')

                json_to_update_push = {k: {'$each': v} for (k,v) in zip(fields_to_update_push, [json[field] for field in fields_to_update_push])}
                mongo_collection.update_one({'_id': label}, {'$push': json_to_update_push})

                json_to_update_set = {k: v for (k,v) in zip(fields_to_update_set, [json[field] for field in fields_to_update_set])}
                mongo_collection.update_one({'_id': label}, {'$set': json_to_update_set})

            else:
                print(f'Already updated until {time_fin.date()}!')
                break


        else: #EL OBJETO DEBE CREARSE
            print('No existe el objeto: se CREA con INSERT')
            for band_name in sel_bands:
                print(band_name)
                func = statistics_function[stat]
                collection = ee.ImageCollection(satellite.collectionId).filterDate(date_ini, date_fin)
                if band_name in vegetation_index.keys():
                    band = band_name
                    collection = vegetation_index[band](collection, bands, parameters)
                else: band = bands[band_name]

                collection = collection.select(band)
                feature_of_interest = ee.Feature(crops_json[i])
                collection_of_interest = func(collection.filterBounds(feature_of_interest.geometry()), band, feature_of_interest.geometry(), scale)
                band_values, dates = StatisticsBand(collection_of_interest, band, feature_of_interest, stat, -0.1)
                json[band_name]=band_values

            str_dates = [dates[i].strftime("%Y-%m-%d") for i in range(len(dates))]
            json['dates']=dates
            json['str_dates']=str_dates
            mongo_collection.insert_one(json)

def updateMultiSeriesMongo(satellite, date_ini, date_fin, crops_json, sel_bands, stat, mongo_collection, parameters={'WDVI_slope': 4, 'SAVI_L': 2}, scale=20):
    if datetime.strptime(date_fin, "%Y-%m-%d") < datetime.strptime(satellite.oldestDate, "%Y-%m-%d"):
        time_ini = satellite.oldestDate
        print(f'Initial date before: changing to {datetime.strptime(satellite.oldestDate, "%Y-%m-%d")}')

    time_fin = datetime.strptime(date_fin, "%Y-%m-%d")
    bands = satellite.bandsDict

    for i, crop in enumerate(crops_json):
        label=crops_json[i]['properties']['id'] + '_' + satellite.name
        json={'_id':label,'field': crops_json[i]['properties']['id'],'satellite':satellite.name, 'type':'index'}
        print('\n', i, label)

        fields_to_update_set = []
        fields_to_update_push = []

        if mongo_collection.find_one({'_id': label}): #EL OBJETO YA EXISTE -> HARE UPDATE
            saved_dates = mongo_collection.find_one({'_id': label})['dates']
            len_dates = len(saved_dates)
            time_ini = saved_dates[-1] + timedelta(days=1)

            # Calculo bandas
            if time_ini.date() < time_fin.date():
                for band_name in sel_bands:
                    print(band_name)
                    if band_name in mongo_collection.find_one({'_id': label}).keys(): #UPDATE
                        len_band = len(mongo_collection.find_one({'_id': label})[band_name])
                        dif_len = len_dates - len_band
                        fields_to_update_push.append(band_name)
                        #print(len_band, len_dates)

                        if dif_len > 0: time_ini = saved_dates[-dif_len]
                        else: time_ini = saved_dates[-1] + timedelta(days=1)

                    else: #UPDATE SET
                        dif_len = len_dates
                        time_ini = saved_dates[0] 
                        fields_to_update_set.append(band_name)
                    
                    str_time_ini = time_ini.strftime("%Y-%m-%d")
                    str_time_fin = time_fin.strftime("%Y-%m-%d")

                    func = statistics_function[stat]
                    collection = ee.ImageCollection(satellite.collectionId).filterDate(str_time_ini, str_time_fin)
                    if band_name in vegetation_index.keys():
                        band = band_name
                        collection = vegetation_index[band](collection, bands, parameters)
                    else: band = bands[band_name]

                    collection = collection.select(band)
                    feature_of_interest = ee.Feature(crops_json[i])
                    collection_of_interest = func(collection.filterBounds(feature_of_interest.geometry()), band, feature_of_interest.geometry(), scale)
                    band_values, dates = StatisticsBand(collection_of_interest, band, feature_of_interest, stat, -0.1)

                    dates_to_update = [date for date in dates if date not in saved_dates]

                    bands_to_update = band_values
                    str_dates_to_update = [dates_to_update[i].strftime("%Y-%m-%d") for i in range(len(dates_to_update))]
                    json[band_name]=bands_to_update

                json['dates']=dates_to_update
                json['str_dates']=str_dates_to_update
                fields_to_update_push.append('dates')
                fields_to_update_push.append('str_dates')

                json_to_update_push = {k: {'$each': v} for (k,v) in zip(fields_to_update_push, [json[field] for field in fields_to_update_push])}
                mongo_collection.update_one({'_id': label}, {'$push': json_to_update_push})

                json_to_update_set = {k: v for (k,v) in zip(fields_to_update_set, [json[field] for field in fields_to_update_set])}
                mongo_collection.update_one({'_id': label}, {'$set': json_to_update_set})

            else:
                print(f'Already updated until {time_fin.date()}!')
                break


        else: #EL OBJETO DEBE CREARSE
            print('No existe el objeto: se CREA con INSERT')
            print(sel_bands)
            for band_name in sel_bands:
                print(band_name)
                func = statistics_function[stat]
                collection = ee.ImageCollection(satellite.collectionId).filterDate(date_ini, date_fin)
                if band_name in vegetation_index.keys():
                    band = band_name
                    collection = vegetation_index[band](collection, bands, parameters)
                else: band = bands[band_name]

                collection = collection.select(band)
                feature_of_interest = ee.Feature(crops_json[i])
                collection_of_interest = func(collection.filterBounds(feature_of_interest.geometry()), band, feature_of_interest.geometry(), scale)
                band_values, dates = StatisticsBand(collection_of_interest, band, feature_of_interest, stat, -0.1)
                json[band_name]=band_values

            json['dates']=dates
            json['str_dates']=[date.strftime("%Y-%m-%d") for date in dates]
            mongo_collection.insert_one(json)

def updateFilterBandsMongo(satellite, crops_id_list, input_collection, output_collection, sel_bands, filter='valleys', smoothing='SG_smoothing'):
    for crop_id in crops_id_list:
        label=f'crop_{crop_id}_{satellite.name}_band'
        json_input = input_collection.find_one({'_id': label})
        json_output = {'_id': label, 'field': f'crop_{crop_id}', 'satellite': satellite.name}

        for band_name in sel_bands:
            if band_name in json_input.keys():
                ts = pd.DataFrame({'dates': json_input['dates'], 'values': json_input[band_name]})
                index = np.array(pd.to_datetime(ts['dates']))
                raw_series = pd.Series(np.array(ts['values']), index = index)

                dt_most_frequent = Counter((raw_series.diff(1).index[1:]-raw_series.diff(1).index[:-1]).days).most_common(1)[0][0]
                original_interpolation_frequency = str(dt_most_frequent) + 'D'
                series = raw_series.resample(original_interpolation_frequency).mean().interpolate(method='linear')

                # FILTRO --------------------------------------------------------------------------------------------------------
                if filter == 'valleys':
                    series_threshold = 0.1
                    filtered_series, valleys = remove_threshold_valley(series, series_threshold, th_proportional=False)
                    if satellite.name == 'sentinel2' and weekly_interpolation == 'yes':
                        print('Weekly interpolation')
                        filtered_series=filtered_series.resample('7D').mean().interpolate(method='linear')

                    dates = filtered_series.index
                    json_output['dates'] = list(dates)
                    json_output['str_dates'] = list(dates.strftime("%Y-%m-%d"))

                # Suavizado ------------------------------------------------------------------------------------------------------
                if smoothing == 'SG_smoothing':
                    #Parametros optimos para obtener la serie Ntr (long-term change) por el filtro SG
                    m, d = optimal_SG_parameters(filtered_series)
                    Ntr = savgol_filter(filtered_series, window_length=m*2+1, polyorder=d)
                    Ntr = pd.Series(Ntr, index=filtered_series.index)
                    #Iteracion de la serie por el filtro SG: La serie final new_series es la que minimiza la función fit effect index
                    ind_min, N1, N, fit_effect = SG_filtering(filtered_series, half_window=m, polyorder=d, num_iterations=5)
                    new_series = pd.Series(N[ind_min], index=filtered_series.index)


                elif smoothing == 'MAXMIN_smoothing':
                    potential_extrema, label = detect_extrema(filtered_series, extrema_window)
                    extrema, label = remove_consecutive_extrema(filtered_series, potential_extrema, label)
                    final_extrema, label = remove_threshold_extrema(filtered_series, extrema, label, extrema_threshold)
                    final_extrema, label = remove_consecutive_extrema(filtered_series, final_extrema, label)    
                    new_series = pd.Series(iterative_filtering(filtered_series, final_extrema), filtered_series.index)
                    print('serie suavizada con método MAXMIN:\n', new_series)

                json_output[band_name] = list(new_series.values)
            else: print(f'{band_name} not found!'); pass;

        print('Saving in Mongo...')
        try:
            if output_collection.find_one({'_id': label}): #Si ya existe el objeto
                output_collection.update_one({'_id': label}, {'$set': json_output})
            else:
                output_collection.insert_one(json_output)
        except: print('Error!'); pass



# Gráfica
def plot_series(satellite, date_ini, date_fin, crop_id=25, stat='MAX', band_name='NDVI', parameters={'WDVI_slope': 4, 'SAVI_L': 2}, scale=20):
    print('Existing bands:', ', '.join(bands.keys()), '\nSelecting:', band_name)

    # Buscamos en la lista de cultivos el que tiene el identificador deseado
    for i in range(len(crops_json)):
        Id = crops_json[i]['properties']['id']
        if Id[5:] == str(crop_id):
            crop_index = i
            break

    func = statistics_function[stat]
    collection = ee.ImageCollection(satellite.collectionId).filterDate(date_ini, date_fin)
    if band_name in vegetation_index.keys():
        band = band_name
        collection = vegetation_index[band](collection, bands, parameters)
    else: band = bands[band_name]

    collection = collection.select(band)
    feature_of_interest = ee.Feature(crops_json[crop_index])
    collection_of_interest = func(collection.filterBounds(feature_of_interest.geometry()), band, feature_of_interest.geometry(), scale)
    band_values, dates = StatisticsBand(collection_of_interest, band, feature_of_interest, stat, -0.1)

    # Gráfica
    plt.figure(figsize=(8, 5))
    plt.xlabel('Days')
    plt.ylabel(stat + ' ' + band + ' (' + band_name + ')')
    plt.plot(dates, band_values, marker='o', markersize=3, \
            label=crops_json[crop_index]['properties']['id'] + ' ' + satellite.name)
    plt.legend()
    plt.show()

class satellite:
    def __init__(self, name, collectionId, oldestDate, bandsDict):
        self.name = name
        self.collectionId = collectionId
        self.oldestDate = oldestDate
        self.bandsDict = bandsDict


def addCLOUDS (collection, cloudsCollection):
    filterTimeEq = ee.Filter.equals(**{'leftField': 'system:index',
                                        'rightField': 'system:index'})
    innerJoinedCol = ee.Join.inner().apply(collection, cloudsCollection, filterTimeEq)
    joinedCol = ee.ImageCollection(innerJoinedCol.map(lambda feature:
                                   ee.Image.cat(feature.get('primary'), feature.get('secondary'))))
    return joinedCol

def countPixels(image, geometry, scale=20): # Cuenta los píxeles a escala scale dentro de la region geometry
    band = image.bandNames().get(0)
    count = image.reduceRegion(geometry = geometry,
                                reducer = ee.Reducer.count(),
                                scale = scale).get(band)
    return count

def maskCloudsL8(image, band='QA_PIXEL'): #Mascara de nubes para Landsat8
    qa = image.select(band);
    maskWater = 1 << 7 # agua
    maskCirrus = 1 << 15 # Cirrus
    maskCoudShadow = 1 << 11 # high confidence cloud shadow eq 0
    maskCloud = 1 << 9 # high confidence cloud eq 0
    mask = qa.bitwiseAnd(maskCirrus).eq(0).And(qa.bitwiseAnd(maskCoudShadow).eq(0))\
                  .And(qa.bitwiseAnd(maskCloud).eq(0))
    return image.updateMask(mask)

def cloudlessPercentageL8(image, geometry, band='QA_PIXEL'):
    def foo(image):
        masked = maskCloudsL8(image, band)
        count_masked = countPixels(masked, geometry)
        count = countPixels(image, geometry)
        qa_cloud_percentage = (ee.Number(count).subtract(ee.Number(count_masked))).divide(ee.Number(count)).multiply(100)
        return image.set('CLOUDS', qa_cloud_percentage)
    if type(image) ==  ee.imagecollection.ImageCollection: 
        return image.map(foo)
    else:
        return foo(image)

def getTimeSeriesProperty(collection, Property):
    if type(collection) ==  ee.image.Image:
        try:
            inds = collection.get(Property).getInfo()
            dates = datetime.fromtimestamp(collection.get('system:time_start').getInfo()/1000).date()
        except:
            print('Error at dt=', datetime.fromtimestamp(image['properties']['system:time_start']/1000))
            pass
    else:
        dates, inds = [], []
        for i, image in enumerate(collection.getInfo()['features']):
            try: 
                ind = image['properties'][Property]
                date = datetime.fromtimestamp(image['properties']['system:time_start']/1000).date()
                if date not in dates:
                    dates.append(date)
                    inds.append(ind)
            except:
                print('Error at dt=', datetime.fromtimestamp(image['properties']['system:time_start']/1000))
                pass
    return dates, inds

# PREPARACIÓN DATOS ---------------------------------------------------------------------
# Preparamos los diccionarios para acceder fácilmente a las funciones estadísticas y satélites
statistics_function = {'MEAN': meanBand, 'MEDIAN': medianBand, 'MIN': minBand, 'MAX': maxBand,
                       'STDEV': stdevBand, 'MODE': modeBand}

vegetation_index = {'NDVI': addNDVI, 'SAVI': addSAVI, 'MSAVI': addMSAVI, #'MCARI': addMCARI,
                    'MCARI2': addMCARI2, 'ARVI': addARVI, 'WDVI': addWDVI, 'TNDVI': addTNDVI,
                    'GNDVI': addGNDVI}


landsat7 = satellite(name='landsat7',
                    collectionId='LANDSAT/LE07/C02/T1_L2',
                    oldestDate='1999-01-01',
                    bandsDict= {'blue': 'SR_B1', 'green': 'SR_B2', 'red': 'SR_B3', 'nir': 'SR_B4', 'swir1': 'SR_B5', 'swir2': 'SR_B7', 'temperature': 'ST_B6'})

landsat8 = satellite(name='landsat8',
                    collectionId='LANDSAT/LC08/C02/T1_L2',
                    oldestDate='2013-04-11',
                    bandsDict= {'ultrablue': 'SR_B1', 'blue': 'SR_B2', 'green': 'SR_B3', 'red': 'SR_B4', 'nir': 'SR_B5', 'swir1': 'SR_B6','swir2': 'SR_B7',
                            'temperature': 'ST_B10'})

sentinel2 = satellite(name='sentinel2',
                    collectionId='COPERNICUS/S2_SR',
                    oldestDate='2017-03-28',
                    bandsDict={'ultrablue': 'B1', 'blue': 'B2', 'green': 'B3', 'red': 'B4', 'red1': 'B5', 'red2': 'B6', 'red3': 'B7', 'nir': 'B8', 'red4': 'B8A', 'water_vapor': 'B9',
                          'swir1': 'B11','swir2': 'B12'})

sentinel2clouds = satellite(name='sentinel2clouds',
                    collectionId='COPERNICUS/S2_CLOUD_PROBABILITY',
                    oldestDate='2017-03-28',
                    bandsDict={'CLOUDS':'probability'})

satellites_dict = {'sentinel2': sentinel2, 'landsat7': landsat7, 'landsat8': landsat8}

"""
# GLOSSARY BANDS
LANDSAT 7
SR_B1: (blue) surface reflectance
SR_B2: (green) surface reflectance
SR_B3: (red) surface reflectance
SR_B4: (near infrared) surface reflectance
SR_B5: (shortwave infrared 1) surface reflectance
SR_B7: (shortwave infrared 2) surface reflectance
ST_B6: brightness temperature

LANDSAT 8∫ß∫ß
SR_B1: ultra blue
SR_B2: blue
SR_B3: green
SR_B4: red
SR_B5: near infrared
SR_B6: shortwave infrared 1
SR_B7: shortwave infrared 2
ST_B10: brightness temperature

SENTINEL
B1: Aerosols/ultrablue
B2: Blue
B3: Green
B4: Red
B5: Red Edge 1
B6: Red Edge 2
B7: Red Edge 3
B8: near infrared
B8A: Red Edge 4
B9: Water vapor
B11: shortwave infrared 1
B12: shortwave infrared 2
QA60:
"""
