# Savitzky–Golay Filter
import numpy as np
import scipy as sp
import pandas as pd
from scipy.signal import savgol_filter
#from scipy.signal import savgol_coeffs
import matplotlib.pylab as plt
from ee_functions import optimal_SG_parameters, SG_filtering, threshold_resampling
plt.style.use('default')


# INPUT ---------------------------------------------------------------------------------------
satellite = 'sentinel2'
crop_id = 1
band = 'NDVI'
stat = 'MAX'
cloud_threshold =20
series_threshold = 0.4
interpolation_frequency = '7D'
path_to_timeseries = f'./Bands_TimeSeries/{satellite}/crop_{crop_id}_{stat}_{band}.csv' # ruta a la serie temporal
path_to_cloudy_pixels_percentage = f'./Bands_TimeSeries/{satellite}/cloudy_pixel_percentage.csv'


#Bandas espectrales
ts = pd.read_csv(path_to_timeseries, sep=';')
index = np.array(pd.to_datetime(ts['dates']))
series = pd.Series(np.array(ts['band']), index = index)

# Serie de nubosidad
clouds_ts = pd.read_csv(path_to_cloudy_pixels_percentage, sep=';')
index_clouds = np.array(pd.to_datetime(clouds_ts['dates']))
clouds = pd.Series(np.array(clouds_ts['inds']), index = index_clouds)

index = pd.Series([ind for ind in index if ind in index_clouds])
series = series[index]

N0 = threshold_resampling(series, clouds, interpolation_frequency, cloud_threshold, series_threshold)

#print(np.array(index[:10])[clouds[:10]>0.1], clouds[:10])
#exit(0)

#Parametros optimos para obtener la serie long-term change por el filtro SG
m, d = optimal_SG_parameters(N0)
print(f'Optimal parameters (m,d) = {(m, d)}')
Ntr = savgol_filter(N0, window_length=m*2+1, polyorder=d)
Ntr = pd.Series(Ntr, index=N0.index)
#Iteracion de la serie por el filtro SG
ind_min, N, fit_effect = SG_filtering(N0, half_window=m, polyorder=d)


#Figuras ----------------------------------------------------------------------------------------------------------------------------
plt.figure('Cloudiness flags', figsize=(10,5))
plt.title('Cloudiness flags')
plt.xlabel('time')
plt.ylabel(f'{stat} NDVI (crop {crop_id})')
plt.plot(index, series, label='Original series')
plt.plot(series[clouds>cloud_threshold], '*', c='red', label=f'Cloudy pixel percentage>{cloud_threshold}%')
plt.legend()

plt.figure('N trend', figsize=(10,5))
plt.title('N trend')
plt.xlabel('time')
plt.ylabel(f'{stat} NDVI (crop {crop_id})')
plt.plot(index, series, label='Original series')
plt.plot(N0.index, N0, 'o-', markersize=2, label=f'N0')
plt.plot(Ntr.index, Ntr, 'o-', markersize=2, label=f'N trend')
plt.legend()


#t = N0.index
plt.figure('SG_Filter', figsize=(10,5))
plt.xlabel('time')
plt.ylabel(f'{stat} NDVI (crop {crop_id})')
plt.plot(series.index, series, '-o', markersize=2, label='N0')
plt.plot(N0.index, N[ind_min], '-o', markersize=2, label=f'N{ind_min+2}')
#plt.plot(t[F>cloudy_threshold], N0[F>cloudy_threshold], '*', color='red', label=f'clouds > {cloudy_threshold}')
plt.legend()


plt.figure('Fit-effect Index')
plt.title('Fit-effect Index')
plt.xlabel('k-th iteration')
plt.ylabel('F(k)')
plt.plot(fit_effect, 'o-', markersize=3)

plt.show()