#!/usr/bin/env python
# coding: utf-8
from ee_functions import *
from ts_functions import *
import pymongo

# INPUT -------------------------------------------------------------------------------------------------------------
# Parámetros para eliminar valles
series_threshold = 0.1
#parámetros para nubes
cloud_threshold = 50 # Elimina imágenes con probability_stat por encima de esa probabilidad de nube

# OPCION DE SUAVIZADO ------------------------------------------------------
smoothing_process = 'SG_smoothing'
#smoothing_process = 'MAXMIN_smoothing'

smoothing_func = {'SG_smoothing': SG_smoothing, 'MAXMIN_smoothing': MAXMIN_smoothing}

# Elección parámetros suavizado MAXMIN
extrema_threshold = 0.1	# Mínima diferencia entre máximo y mínimo
extrema_window = 9  # Ventana temporal para detectar los máximos y mínimos
iteration_threshold = 0.025 # Si entre una iteracion y otra el valor de la banda de un dia no cambia mas de este valor mantenemos el valor anterior
maxmin_params = [extrema_threshold, extrema_window, iteration_threshold]

weekly_interpolation = 'y' # De momento solo valido para sentinel2 "yes" activa la opcion

## conect to mongo
client = pymongo.MongoClient(port=27017)
db = client['bigcropdata']
col_input='remoteSensing2'
col_output='smoothIndex'
#--------------------------------------------------------------------------------------------------------------------

smoothing_func = smoothing_func[smoothing_process]

for json_input in db[col_input].find({},{}):  #Vamos iterando sobre cada objeto de la BD
	label = json_input['_id']
	dates_input = np.array([date.date() for date in json_input['dates']])
	satellite = satellites_dict[json_input['satellite']]
	json_output_to_update = {}
	print('\n', label)

	if len(dates_input) > 20: #Se necesitan suficientes datos >20 para hacer el filtro y suavizado
		last_date_input = dates_input[-1]
		sel_bands = [field for field in json_input.keys() if field in list(satellite.bandsDict) + list(vegetation_index.keys())]
		json_output_to_update = {k:json_input[k] for k in json_input.keys() if k not in sel_bands + ['str_dates', 'dates']}
		for band_name in sel_bands:
			if len(dates_input) != len(json_input[band_name]):
				print(f'Need to UPDATE {col_input}-{label}-{band_name} data!')
				pass
			else:
				series = pd.Series(json_input[band_name], index=json_input['dates'])
				interp_series = resampling_interpolation(series)	# Resampleamos e interpolamos
				dates = interp_series.index

				# APLICAMOS FILTRO VALLES
				filtered_series, valleys = remove_threshold_valley(interp_series, series_threshold, th_proportional=False)
				if satellite.name == 'sentinel2' and weekly_interpolation == 'yes':
					filtered_series=filtered_series.resample('7D').mean().interpolate(method='linear')
					dates = filtered_series.index
					
				#APLICAMOS SUAVIZADO
				if smoothing_process=='MAXMIN_smoothing': new_series, _ = smoothing_func(filtered_series, *maxmin_params)
				else: new_series, _ = smoothing_func(filtered_series)

				json_output_to_update[band_name] = list(new_series.values)
				print(band_name)
			json_output_to_update['dates'] = list(dates) # Las fechas las insertaremos en lugar de actualizar, por si hay alguna banda incompleta
			json_output_to_update['str_dates'] = [date.strftime("%Y-%m-%d") for date in dates]
			try: db[col_output].insert_one(json_output_to_update) # crea el campo
			except: db[col_output].update_one({'_id': label}, {'$set': json_output_to_update}) # SET actualiza reescribiendo el campo

	else: print('Not enough data!')
