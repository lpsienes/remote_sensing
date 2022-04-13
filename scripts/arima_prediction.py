import matplotlib.pylab as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import json
import numpy as np

# INPUT --------------------------------------------------------------------------------------------------------
crop_id = 76
stat = 'MAX' # estadístico
band = 'NDVI'  # banda o indice para analizar
satellite = 'sentinel2'  #satelite
interpolation_freq = '7D'  # frecuencia datos (7D=semanal)
n_predictions = 8  # numero de predicciones
path_to_timeseries = f'./Bands_TimeSeries/{satellite}/crop_{crop_id}_{stat}_{band}.csv' # ruta a la serie temporal
path_to_pivot_by_number = './pivot_by_number.json'  #ruta al json que almacena el tipo de cultivo por parcela


'''
   'silage': [1, 20, 43, 49, 50, 53, 64, 72, 74, 95, 121, 122, 123, 124],
         'sugar beat': [2, 6],
         'wheat': [22, 23, 55, 59, 76, 81, 84, 85, 86, 90, 125, 126, 127, 128, 129, 130, 131, 132, 133, 136],
'''


# Functions--------------------------------------------------------------------------------------------------------

def get_MAX_NDVI_prediction(series, type_pivot, n_predictions=8, interpolation_freq = '7D', training=False, plot=True):
	# series: series with maximum values of NDVI of a pivot
	#type_pivot: wheat, silage, sugar beat, scallion,peanut
	#interpolation_freq: 7D = 7 days, 1T = 1 min, 30S = 30 seg, etc
	# pdq optimos segun el tipo de cultivo para MAX NDVI
	if type_pivot == 'Silage': order = (2, 0, 0)
	elif type_pivot == 'Wheat': order = (2, 0, 0)
	elif type_pivot == 'Sugar beat': order = (3, 0, 0)
	else: type_pivot == 'Other'; order = (2, 0, 0)

	resample = series.resample(interpolation_freq).mean().interpolate()
	train = resample

	if training:
		resample = series.resample(interpolation_freq).mean().interpolate()
		test = resample[-n_predictions:]
		train = resample[:-n_predictions]

	model = ARIMA(train, order=order).fit()
	prediction = model.forecast(steps=n_predictions)

	if plot:
		"""
		plt.figure('Fig: prediction timeseries')
		plt.title(f'MAX NDVI: crop {crop_id} ({type_pivot})')
		plt.plot(series, color='tab:blue', label='timeseries')
		plt.plot(resample, color='tab:orange', label='resample')
		plt.plot(prediction, color='tab:green', label='prediction')
		plt.legend()
		"""
		if training:
			fig, ax = plt.figure(f'pred{crop_id}'), plt.axes()
			plt.title(f'{stat} NDVI: crop {crop_id} ({type_pivot})')
			plt.plot(test, marker='o', color='tab:orange', label='test')
			plt.plot(prediction,  marker='o', color='tab:green', label='prediction')
			ax.set_xticks(test.index[-n_predictions-1:].date)
			ax.set_xticklabels(test.index[-n_predictions-1:].date, rotation=20)
			plt.ylim(0,1)
			plt.legend()
			print(model.summary())
	print(test)
	print(test-prediction)
	print(abs(test-prediction))
	print(np.mean(abs(test-prediction)))
	return(prediction)


# Program #-------------------------------------------------------------------------------------------------------
ts = pd.read_csv(path_to_timeseries, sep=';')
pivot_by_number = json.load(open(path_to_pivot_by_number))
type_pivot = 'Other'
for crop in pivot_by_number:
	if crop['crop_id'] == crop_id: type_pivot = crop['crop']

index = np.array(pd.to_datetime(ts['dates']))
series = pd.Series(np.array(ts['band']), index = index)

prediction = get_MAX_NDVI_prediction(series=series, type_pivot=type_pivot, n_predictions=n_predictions,
									 interpolation_freq = interpolation_freq, training=True, plot=True)
print(prediction)

plt.show()
