# TIME SERIES FUNCTIONS
import pandas as pd
from datetime import datetime
from copy import deepcopy
from collections import Counter
import numpy as np
from scipy.signal import savgol_filter


def resampling_interpolation(series):
	dt_most_frequent = Counter((series.diff(1).index[1:]-series.diff(1).index[:-1]).days).most_common(1)[0][0]
	original_interpolation_frequency = str(dt_most_frequent) + 'D'
	interp_series = series.resample(original_interpolation_frequency).mean().interpolate(method='linear')
	return interp_series

# Divide una serie temporal en ciclos anuales de septiembre a septiembre
def annual_splitting_ts (timeseries, dates):
	ts = pd.Series(timeseries, index=pd.to_datetime(dates))
	start_year = pd.to_datetime(ts.index[0]).year
	ts_year = {i: ts[(ts.index >= datetime(start_year + i, 9, 1)) #establecemos el inicio de la temporada en 01/09/YYYY
						  & (ts.index < datetime(start_year + i + 1, 9, 1))] for i 
					  in range(ts.index[-1].year-ts.index[0].year)}
	return ts_year

# Funciones para eliminar puntos --------------------------------------------------------------------------------------
# Elimina valles donde la caída y posterior subida supere un umbral
# Series_threshold: número entre 0, 1 que define el umbral
# th_proportional == True extrapola el umbral a imágenes que distan más del intervalo determinado
#				  == False no tiene encuenta la distancia temporal para eliminar los valles
def remove_threshold_valley(series, series_threshold, th_proportional = False):
	s = deepcopy(series)
	j = 0
	c = 1
	points_to_remove = 0
	dt_most_frequent = Counter((s.diff(1).index[1:]-s.diff(1).index[:-1]).days).most_common(1)[0][0]
	daily_threshold = series_threshold/dt_most_frequent
	while c > 0:
		j += 1
		#print(f'Iter {j}')
		c = 0
		ds = s.diff(1)
		for i in range(len(ds)-1):
			th = series_threshold
			# Calculamos la distancia temporal y el threshold para ese dt
			if th_proportional:
				th = (ds.index[i+1]-ds.index[i]).days*daily_threshold
			if ((ds[i+1] > th) & (ds[i] < -th)):
				s[i] = np.nan # Asignamos nan = Not a Number al valle
				points_to_remove += 1
		if points_to_remove > 0:
			c += 1
			points_to_remove = 0
	valleys = series[np.isnan(s)]
	s = s.interpolate('linear')
	return(s, valleys)

# Elimina puntos con nubosidad superior a un umbral
# series: la serie temporal de la banda a anlizar
# clouds_series: la serie temporal de la banda probabilidad de nubes
# clouds_threshold: porcentaje umbral (valores de 0 a 100)
def remove_threshold_clouds(series, clouds_series, cloud_threshold):
	s = deepcopy(series)
	cloudy_days = s[clouds_series>cloud_threshold]
	s[clouds_series>cloud_threshold] = np.nan
	s = s.interpolate('linear')
	return(s, cloudy_days)

# Funciones de suavizado --------------------------------------------------------------------------------------
######################### Filtro de MÁXIMOS Y MÍNIMOS #########################################################

# Detecta todos los puntos extremos locales en una ventana temporal determinada
def detect_extrema(series, window=9):
	hlfw = int(window/2)
	maxima = np.array([], dtype='int64')
	minima = np.array([], dtype='int64')
	extrema = np.array([], dtype='int64')
	label = []
	for i in range(hlfw, len(series)-hlfw): # antes ponia for i in range(hlfw, len(series)-hlfw+1)
		series_w = series[i-hlfw:i+hlfw+1]
		if series[i] in [max(series_w), min(series_w)]:
			extrema = np.append(extrema, i)
			if series[i] == max(series_w): label.append(1)
			else: label.append(0)
	label = np.array(label)
	return(extrema, label)

# Si existen máximos o mínimos consecutivos elimina uno de ellos quedándose con el mayor de los máximos o el menor de los mínimos
# Series: serie original de los valores de la banda
# extrema: índice de los puntos extremos calculados en la función detect_extrema
def remove_consecutive_extrema(series, extrema, label):
    final_extrema = list(extrema)
    final_label = list(label)
    for i,ext in enumerate(extrema[:-2]):
        if (label[i]==label[i+1]==1):
            if series[extrema[i]] < series[extrema[i+1]]:
                final_extrema.remove(extrema[i])
                final_label.remove(label[i])

        if (label[i]==label[i+1]==0):
            if series[extrema[i]] > series[extrema[i+1]]:
                final_extrema.remove(extrema[i])
                final_label.remove(label[i])
    return(np.array(final_extrema), np.array(final_label))

# Elimina máximos y mínimos locales consecutivos cuya diferencia no llega a un umbral threshold determinado
# label: vector de 0s y 1s que etiqueta los extremos como mínimos (0) o máximos (1)
def remove_threshold_extrema(series, extrema, label, threshold=0.1):
	final_extrema = extrema[~(abs(series[extrema].diff(1))<threshold)]
	label = label[~(abs(series[extrema].diff(1))<threshold)]
	return(final_extrema, label)

# Proceso iterativo de suavizado de la serie cambiando los pesos en cada iteración. El proceso termina cuando no hay ningún
# punto que se vea modificado en +-iteration_threshold unidades de una iteración a la siguiente
def iterative_filtering(series, extrema, iteration_threshold=0.025):
	N_old = deepcopy(series)
	count = 1
	k = 0
	while count != 0:
		k += 1
		filter_weights = np.array([1/(k+2), k/(k+2), 1/(k+2)])
		N_new = np.zeros(len(N_old))
		N_new[0] = N_old[0]
		N_new[-1] = N_old[-1]
		count = 0
		for i in range(1, len(series)-1):
			if i not in extrema:
				if (abs(sum(N_old[i-1:i+2]*filter_weights) - N_old[i]) > abs(iteration_threshold)):
					N_new[i] = sum(N_old[i-1:i+2]*filter_weights)
					count += 1
				else: N_new[i] = N_old[i]
		N_new[extrema] = deepcopy(series[extrema])
		N_old = N_new
 #   print(k, 'iterations')
	return(N_new)

############################### Filtro de SAVITZKY-GOLAY #########################################################
def optimal_SG_parameters(timeseries, half_windows=[7, 6, 5, 4], orders=[2,3,4]):
	least_squares = []
	parameters_l = []
	for window in half_windows:
		for order in orders:
			SG_filter = savgol_filter(timeseries, window_length=window*2+1, polyorder=order)
			least_squares.append(np.sum((SG_filter-timeseries)**2))
			parameters_l.append((window, order))
	parameters = parameters_l[least_squares.index(min(least_squares))]
	return parameters   # Devuelve los parámetros (m,d) optimos por ajuste de minimos cuadrados

def SG_filtering(N0, half_window=4, polyorder=6, num_iterations=10):
	m = half_window; d=polyorder  #Minimizamos m y maximizamos d de entre los valores medios
	Ntr = savgol_filter(N0, 2*m+1, d) #Calculamos la serie long-term change trend y la limitamos a [0,1]

	W = np.array([1-abs(N0[i]-Ntr[i])/max(abs(N0-Ntr)) for i in range(len(N0))])  #Calculamos los pesos de cada punto
	N1 = np.array([max(N0[i], Ntr[i]) for i in range(len(N0))])

	N, F = [], []
	m, d = 4, 6  # Recomendado para las iteraciones siguientes
	N_prev = N1
	for c in range(0, num_iterations):
		N.append(savgol_filter(N_prev, 2*m+1, d))
		F.append(np.sum(abs(N[-1]-N0)*W))
		N_prev = N[-1]
	ind_min = F.index(min(F))
	return ind_min, N1, N, F    # Devuelve el indice de la iteración con menor valor de Fitting-effect
							# la lista de la iteracion de series correspondientes y la serie de Fitting-effect


# SUAVIZADO SG
def SG_smoothing(series):
	m, d = optimal_SG_parameters(series)
	Ntr = savgol_filter(series, window_length=m*2+1, polyorder=d)
	Ntr = pd.Series(Ntr, index=series.index)

	#Iteracion de la serie por el filtro SG: La serie final new_series es la que minimiza la función fit effect index
	ind_min, N1, N, fit_effect = SG_filtering(series, half_window=m, polyorder=d, num_iterations=5)
	new_series = pd.Series(N[ind_min], index=series.index)
	return new_series, [N1, N, fit_effect]

def MAXMIN_smoothing(series, extrema_window=9, extrema_threshold=0.1, iteration_threshold=0.025):
	potential_extrema, label = detect_extrema(series, extrema_window)
	extrema, label = remove_consecutive_extrema(series, potential_extrema, label)
	final_extrema, label = remove_threshold_extrema(series, extrema, label, extrema_threshold)
	final_extrema, label = remove_consecutive_extrema(series, final_extrema, label)    
	new_series = pd.Series(iterative_filtering(series, final_extrema, iteration_threshold), series.index)
	return new_series, [final_extrema, label]




