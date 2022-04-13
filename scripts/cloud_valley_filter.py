import numpy as np
import scipy as sp
import pandas as pd
from scipy.signal import savgol_filter
from collections import Counter
from datetime import datetime
from copy import deepcopy
import matplotlib.pylab as plt
from ts_functions import *
from ee_functions import sentinel2, landsat7, landsat8, vegetation_index

# INPUT ##############################################################################################################################
year = 0  # El índice del año que se quiere estudiar.
satellite = landsat8
crop_id = 1		# El identificador del cultivo
band_name = 'blue'
stat = 'MAX'
# OPCION FILTRO: ------------------------------------------------------
# Se elige uno de los dos
series_filter = 'CLOUDS' # Este filtro todavia no esta implementado para landsat7
#series_filter = 'VALLEYS'
# En caso de elegir sentinel2 y series_filter='CLOUDS' hay que establecer el estadistico de prob
probability_stat = 'MAX'
# Parámetros para eliminar valles
series_threshold = 0.1
#parámetros para nubes
cloud_threshold = 50 # Elimina imágenes con probability_stat por encima de esa probabilidad de nube

# OPCION DE SUAVIZADO ------------------------------------------------------
# Se pueden mostrar ambas a la vez
SG_smoothing = 'yes'
MAXMIN_smoothing = 'yes'
# Elección parámetros suavizado MAXMIN
extrema_threshold = 0.15	# Mínima diferencia entre máximo y mínimo
extrema_window = 9  # Ventana temporal para detectar los máximos y mínimos
iteration_threshold = 0.05 # Si entre una iteracion y otra el valor de la banda de un dia no cambia mas de este valor mantenemos el valor anterior

if band_name not in vegetation_index.keys():
	band = satellite.bandsDict[band_name]
else: band = band_name
# Ruta a las series de la bandas del índice de vegetación y banda de probabilidad de nubes ---------------------------------------------
path_to_timeseries = f'./Bands_TimeSeries/{satellite.name}/{crop_id}_{stat}_{band}.csv' # ruta a la serie temporal
if series_filter == 'CLOUDS':
	if satellite.name == 'sentinel2':
		path_to_cloud_probability = f'./Bands_TimeSeries/{satellite.name}/{crop_id}_{probability_stat}_probability.csv' # ruta a la serie temporal
	if satellite.name == 'landsat8':
		path_to_cloud_probability = f'./Bands_TimeSeries/{satellite.name}/{crop_id}_CLOUDS.csv' # ruta a la serie temporal

weekly_interpolation = 'yes' # Si queremos tener la serie interpolada semanalmente
write_filtered_series = 'yes' # Escribir 'yes' si queremos escribir las series tras el filtro
write_smoothed_series = 'yes' # Escribir 'yes' si queremos escribir las series tras el suavizado
file_to_write = './Bands_TimeSeries/' + satellite.name + '/' + str(crop_id) + '_' + stat + '_' + band + '_' + series_filter
##########################################################################################################################################


# PREPARACION DATOS ##############################################################################################################################
#Bandas espectrales
ts = pd.read_csv(path_to_timeseries, sep=';')
index = np.array(pd.to_datetime(ts['dates']))
raw_series = pd.Series(np.array(ts['values']), index = index)


# Serie de porcentaje de pixels de nubosidad
if series_filter == 'CLOUDS':
	clouds_ts = pd.read_csv(path_to_cloud_probability, sep=';')
	index_clouds = np.array(pd.to_datetime(clouds_ts['dates']))
	clouds = pd.Series(np.array(clouds_ts['values']), index = index_clouds)
	clouds_by_year = annual_splitting_ts(clouds.values, clouds.index)
series_by_year = annual_splitting_ts(raw_series.values, raw_series.index)
years = [series_by_year[i].index[0].year for i in range(len(series_by_year))]


# PROGRAMA ##############################################################################################################################
series_by_year_interpolated, clouds_by_year_interpolated = {}, {}
for y in range(len(series_by_year)):
	series = series_by_year[y]
	dt_most_frequent = Counter((series.diff(1).index[1:]-series.diff(1).index[:-1]).days).most_common(1)[0][0]
	original_interpolation_frequency = str(dt_most_frequent) + 'D'
	series = series.resample(original_interpolation_frequency).mean().interpolate(method='linear')
	series_by_year_interpolated[y] = series


if series_filter == 'CLOUDS':
	for y in range(len(series_by_year)):
		clouds = clouds_by_year[y]
		clouds = clouds.resample(original_interpolation_frequency).mean().interpolate(method='linear')
		index_clouds = clouds.index
		series = series_by_year[y]
		index_series = series.index
		index = pd.Series(np.array([ind for ind in index_series if ind in index_clouds]))
		series = series[index]
		clouds = clouds[index]
		clouds_by_year_interpolated[y] = clouds
		clouds_series = clouds_by_year[year]
		series_by_year_interpolated[y] = series

series = series_by_year_interpolated[year]

# Filtro valle
if series_filter == 'VALLEYS':
	filtered_series, valleys = remove_threshold_valley(series, series_threshold, th_proportional=False)
	print('serie filtrada por valles:\n', filtered_series)

	plt.figure(figsize=(8,4))
	plt.title('Valley filter')
	plt.ylabel(f'{stat} NDVI (crop {crop_id} {years[year]})')
	plt.xlabel('time')
	plt.xticks(fontsize=9, rotation=0)
	plt.plot(series, '--o', linewidth=0.75, color ='tab:blue', label=f'raw series {satellite.name}')
	plt.plot(filtered_series, '-o', linewidth=1.25, color ='tab:blue',  label='filtered series',)
	plt.plot(valleys, 'x', color='red', label=f'removed with h={series_threshold}')
	plt.legend()
	
	if write_filtered_series == 'yes':  # Escribimos en una carpeta un .csv la serie temporal, p. ej: "./Bands_Timeseries/sentinel2/crop_21_MEAN_NDVI_VALLEY_filtered.csv" 
		df = pd.DataFrame({'dates': filtered_series.index, 'band': filtered_series.values})
		df.to_csv(file_to_write + '_' + str(years[year]) + '.csv',
	 							index=False, sep=';')


# Filtro nube
if series_filter == 'CLOUDS':
	clouds_series = clouds_by_year_interpolated[year]
	filtered_series, clouds_above_th = remove_threshold_clouds(series, clouds_series, cloud_threshold)
	print('serie filtrada por nubes:\n', filtered_series)

	plt.figure(figsize=(8,4))
	plt.title('Cloud filter')
	plt.ylabel(f'{probability_stat} {band} (crop {crop_id} {years[year]})')
	plt.xlabel('time')
	plt.plot(series, '--o', linewidth=0.75, color ='tab:blue',  label=f'raw series {satellite.name}')
	plt.plot(filtered_series, '-o', linewidth=1.25, color ='tab:blue',  label='filtered series')
	plt.plot(clouds_above_th, 'x', c ='red', label=f'{probability_stat} cloud prob > {cloud_threshold} %')
	plt.legend()
	
	if write_filtered_series == 'yes':  # Escribimos en una carpeta un .csv la serie temporal, p. ej: "./Bands_Timeseries/sentinel2/crop_21_MEAN_NDVI_CLOUDS_filtered.csv" 
		df = pd.DataFrame({'dates': filtered_series.index, 'band': filtered_series.values})
		df.to_csv(file_to_write + '_' + str(years[year])  + '.csv',
	 							index=False, sep=';')

if weekly_interpolation == 'yes': filtered_series=filtered_series.resample('7D').mean().interpolate(method='linear')

# Suavizado
if SG_smoothing == 'yes':
	#Parametros optimos para obtener la serie Ntr (long-term change) por el filtro SG
	m, d = optimal_SG_parameters(filtered_series)
	Ntr = savgol_filter(filtered_series, window_length=m*2+1, polyorder=d)
	Ntr = pd.Series(Ntr, index=filtered_series.index)

	#Iteracion de la serie por el filtro SG: La serie final new_series es la que minimiza la función fit effect index
	ind_min, N1, N, fit_effect = SG_filtering(filtered_series, half_window=m, polyorder=d, num_iterations=5)
	new_series = pd.Series(N[ind_min], index=filtered_series.index)
	print('serie suavizada con método Savitzky-Golay:\n', new_series)
	plt.figure('SG smoothing', figsize=(8, 4))
	plt.title('SG smoothing')
	plt.xlabel('time')
	plt.ylabel(f'{stat} {band} (crop {crop_id} {years[year]})')
	plt.plot(filtered_series.index, filtered_series, '-o', markersize=4, color='tab:blue', linewidth=0.75, label=f'Original series {satellite.name}')
	#plt.plot(filtered_series.index, Ntr, '-o', markersize=4, linewidth=1.25, color='tab:orange', label='Trend series')
	plt.plot(filtered_series.index, new_series, '-o', markersize=4, linewidth=1.25, color='tab:green', label='SG smoothed series')
	plt.legend()
	if write_smoothed_series == 'yes':  # Escribimos en una carpeta un .csv la serie temporal, p. ej: "./Bands_Timeseries/sentinel2/crop_21_MEAN_NDVI.csv"
		df = pd.DataFrame({'dates': new_series.index, 'band': new_series.values})
		df.to_csv(file_to_write + '_SG_' + str(years[year])  + '.csv',
	 							index=False, sep=';')

if MAXMIN_smoothing == 'yes':
	potential_extrema, label = detect_extrema(filtered_series, extrema_window)
	extrema, label = remove_consecutive_extrema(filtered_series, potential_extrema, label)
	final_extrema, label = remove_threshold_extrema(filtered_series, extrema, label, extrema_threshold)
	final_extrema, label = remove_consecutive_extrema(filtered_series, final_extrema, label)    
	new_series = pd.Series(iterative_filtering(filtered_series, final_extrema), filtered_series.index)
	print('serie suavizada con método MAXMIN:\n', new_series)

	plt.figure('Max/min smoothing', figsize=(8, 4))
	plt.title('Max/min smoothing')
	plt.xlabel('time')
	plt.ylabel(f'{stat} NDVI (crop {crop_id} {years[year]})')
	plt.plot(filtered_series, '-o', markersize=4, linewidth=0.75, color='tab:blue', label=f'Original series {satellite.name}')
	plt.plot(filtered_series.index, new_series, '-o', markersize=4, color='tab:green', label='Max/min smoothed series')
	plt.legend()

	if write_smoothed_series == 'yes':  # Escribimos en una carpeta un .csv la serie temporal, p. ej: "./Bands_Timeseries/sentinel2/crop_21_MEAN_NDVI.csv"
		df = pd.DataFrame({'dates': new_series.index, 'band': new_series.values})
		df.to_csv(file_to_write + '_MAXMIN_' + str(years[year]) + '.csv',
		 						index=False, sep=';')
plt.show()


