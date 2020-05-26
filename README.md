# Forecast Ventas
Proyecto para la clase de mineria de datos de la Universidad del Valle, para realizar una predicción de ventas netas utilizando un forecast de series de tiempo, siendo este una secuencia de observaciones de una variable que se mide en puntos sucesivos del tiempo o en periodos determinados. Donde se exploran los datos en una gráfica de serie de tiempo para identificar su patrón. Esto debido a que las condiciones de un negocio pueden cambiar en el tiempo, haciendo que el patrón de la serie también cambie (Anderson et al., 2016).

Por lo que validaremos una predicción de serie de tiempo para una empresa guatemalteca de Retail, utilizando la Liberia estadística SARIMAX de statsmodels. Debemos de tomar en cuenta que en los modelos de serie de tiempo es muy común utilizar el método de promedios móviles, el cual utiliza los valores de los datos más recientes para pronosticar una serie de tiempo del periodo siguiente. En el que SARIMAX utiliza tres parámetros que explican la estacionalidad, tendencia y ruido de los datos [ARIMA(p,d,q)] (Anderson *et al.*, 2016; Li, 2018).


# Autores 
* Pablo Sao 
* María Mercedes Retolaza

Para correr el programa: python forecast.py 

# Referencias
Anderson, D., Sweeney, D., Williams, T., Camm, J. y Cochran, J. (2016). *Estadística para Negocios y economía. 12va edición*. Ciudad de México, México: CENGAGE. 800 – 817 pp.

Brownlee, J. (2018). *A gentle Introduction to SARIMA for Time Series Forecasting in Python*. Extraído de: https://machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/

Li, S. (2018). *An end-to-end Project on Time Series Analysis and Forecasting with Python*. Extraído de: https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b


