import ee
import folium
import webbrowser

def get_s2_sr_cloud_collection(aoi, start_date, end_date, cloud_filter):
    # Import and filter S2 SR.
    s2_sr_collection = ee.ImageCollection('COPERNICUS/S2_SR')\
                        .filterBounds(aoi).filterDate(start_date, end_date)\
                        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_filter))

    # Import and filter s2cloudless.
    s2_cloudless_collection = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')\
                                .filterBounds(aoi).filterDate(start_date, end_date)

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_collection,
        'secondary': s2_cloudless_collection,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })}))
                              
def add_cloud_bands(image, cloud_probability_threshold):
    def foo(image):
        # Get s2cloudless image, subset the probability band.
        cloud_probability = ee.Image(image.get('s2cloudless')).select('probability')

        # Condition s2cloudless by the probability threshold value.
        is_cloud = cloud_probability.gt(cloud_probability_threshold).rename('clouds')

        # Add the cloud probability layer and cloud mask as image bands.
        return image.addBands(ee.Image([cloud_probability, is_cloud]))
    
    if type(image) ==  ee.imagecollection.ImageCollection: 
        return image.map(foo)
    else:
        return foo(image)


def add_shadow_bands(image, nir_dark_threshold, cloud_projection_distance):
    def foo(image):
        # Identify water pixels from the SCL band.
        not_water = image.select('SCL').neq(6)

        # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
        sr_band_scale = 1e4
        dark_pixels = image.select('B8').lt(nir_dark_threshold*sr_band_scale).multiply(not_water).rename('dark_pixels')

        # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
        shadow_azimuth = ee.Number(90).subtract(ee.Number(image.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

        # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
        cloud_projection = (image.select('clouds').directionalDistanceTransform(shadow_azimuth, cloud_projection_distance*10)
            .reproject(**{'crs': image.select(0).projection(), 'scale': 100})
            .select('distance')
            .mask()
            .rename('cloud_transform'))

        # Identify the intersection of dark pixels with cloud shadow projection.
        shadows = cloud_projection.multiply(dark_pixels).rename('shadows')

        # Add dark pixels, cloud projection, and identified shadows as image bands.
        return image.addBands(ee.Image([dark_pixels, cloud_projection, shadows]))

    if type(image) ==  ee.imagecollection.ImageCollection: 
        return image.map(foo)
    else:
        return foo(image)


def add_cloud_shadow_mask(image, cloud_probability_threshold, nir_dark_threshold, cloud_projection_distance,
                         buffer):
    def foo(image):
        # Add cloud component bands.
        img_cloud = add_cloud_bands(image, cloud_probability_threshold)

        # Add cloud shadow component bands.
        img_cloud_shadow = add_shadow_bands(img_cloud, nir_dark_threshold, cloud_projection_distance)

        # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
        is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

        # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
        # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
        is_cld_shdw = (is_cld_shdw.focal_min(2).focal_max(buffer*2/20)
            .reproject(**{'crs': image.select([0]).projection(), 'scale': 20})
            .rename('cloudmask'))

        # Add the final cloud-shadow mask to the image.
        return img_cloud_shadow.addBands(is_cld_shdw)
    
    if type(image) ==  ee.imagecollection.ImageCollection: 
        return image.map(foo)
    else:
        return foo(image)

# Define a method for displaying Earth Engine image tiles to a folium map.
def add_ee_layer(self, ee_image_object, vis_params, name, show=True, opacity=1, min_zoom=0):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        show=show,
        opacity=opacity,
        min_zoom=min_zoom,
        overlay=True,
        control=True
        ).add_to(self)

def display_cloud_layers(col, aoi, save=False):
    # Mosaic the image collection.
    image = col.mosaic()

    # Subset layers and prepare them for display.
    clouds = image.select('clouds').selfMask()
    shadows = image.select('shadows').selfMask()
    dark_pixels = image.select('dark_pixels').selfMask()
    probability = image.select('probability')
    cloudmask = image.select('cloudmask').selfMask()
    cloud_transform = image.select('cloud_transform')

    # Create a folium map object.
    center = aoi.centroid(10).coordinates().reverse().getInfo()
    m = folium.Map(location=center, zoom_start=12)

    # Add layers to the folium map.
    m.add_ee_layer(image,
                   {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 2500, 'gamma': 1.1},
                   'S2 image', True, 1, 9)
    m.add_ee_layer(probability,
                   {'min': 0, 'max': 100},
                   'probability (cloud)', False, 1, 9)
    m.add_ee_layer(clouds,
                   {'palette': 'e056fd'},
                   'clouds', False, 1, 9)
    m.add_ee_layer(cloud_transform,
                   {'min': 0, 'max': 1, 'palette': ['white', 'black']},
                   'cloud_transform', False, 1, 9)
    m.add_ee_layer(dark_pixels,
                   {'palette': 'orange'},
                   'dark_pixels', False, 1, 9)
    m.add_ee_layer(shadows, {'palette': 'yellow'},
                   'shadows', False, 1, 9)
    m.add_ee_layer(cloudmask, {'palette': 'orange'},
                   'cloudmask', True, 0.5, 9)

    # Add a layer control panel to the map.
    m.add_child(folium.LayerControl())

    # Save the map.
    if save:
        m.save('map.html')
    return m


def open_browser(file, browser='firefox'):
    # MacOS
    if browser=='chrome': path = 'open -a /Applications/Google\ Chrome.app %s'
    else: path = 'open -a /Applications/Firefox.app %s'


    # Windows
    # chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'

    # Linux
    # chrome_path = '/usr/bin/google-chrome %s'

    return(webbrowser.get(path).open(file))