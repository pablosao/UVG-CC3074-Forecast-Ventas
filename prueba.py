import sm as sm

import datamanager
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
warnings.warn("deprecated", DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib

from datetime import datetime

DATA_SERIE = datamanager.getFechasVentasProducto('017027')

#print(DATA_SERIE.sort_values(by=['fecha']).head(5)['fecha'].tolist()[0])
min_date = DATA_SERIE.sort_values(by=['fecha']).head(5)['fecha'].tolist()[0]




DATA_SERIE = DATA_SERIE.set_index('fecha')

#print(DATA_SERIE.head())

#print( int(DATA_SERIE['ventas'].count()/2) )

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(DATA_SERIE,freq = 29,model='additive')

fig = decomposition.plot()
plt.show()

DATA_SERIE.plot(figsize=(15, 6))
plt.show()
###################################################################




p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
# print('Examples of parameter combinations for Seasonal ARIMA...')
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))




df = pd.DataFrame(columns=['param','param_seasonal','AIC'])
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:

            mod = sm.tsa.statespace.SARIMAX(DATA_SERIE,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            trend='ct',
                                            enforce_stationarity=False,
                                            enforce_invertibility=False,
                                            measurement_error = True)
            results = mod.fit()
            df = df.append({'param': param,'param_seasonal':param_seasonal,'AIC':results.aic}, ignore_index=True)
            #print('ARIMA{0}x{1}12 - AIC:{2}'.format(param, param_seasonal, results.aic))
        except:
            continue

df = df.sort_values(by=['AIC']).head(5)
#print(df)
print('Param: '+ str(df['param'].tolist()[0]))
print('seasonal: '+ str(df['param_seasonal'].tolist()[0]))

mod = sm.tsa.statespace.SARIMAX(DATA_SERIE,
                                order=df['param'].tolist()[0],
                                seasonal_order=df['param_seasonal'].tolist()[0],
                                trend='t',
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                                measurement_error = True)

results = mod.fit()
#print(results.summary().tables[1])

start_year = min_date.year

pred = results.get_prediction(start=pd.to_datetime(pd.to_datetime('2014-01-01')), dynamic=False)
pred_ci = pred.conf_int()

pred_ci['diferencia'] = pred_ci['upper ventas'] - pred_ci['lower ventas']



ax = DATA_SERIE[min_date:].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.8, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.3)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()




pred2 = pred.predicted_mean.apply(np.exp)
pred2 = pred2.to_frame()

fechas=(pred2.index).to_list()
print(fechas)
print(pred2[0].to_list())
print(list(pred2.columns))

print()

print(pred_ci.head())

import plotly.graph_objs as go
import plotly.express as px
