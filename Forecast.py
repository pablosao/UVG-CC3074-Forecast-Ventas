"""
Aplicación de Dash con el análisis de prediccion de ventas para un producto
:author: Pablo Sao
:date: 30 de mayo de 2020
"""

import subprocess
import sys

# Importamos libreria necesaria para ejecutar programa
try:

    import pyodbc

except ImportError:
    print("No posee la libreria 'pymssql'. El programa se encuentra instalandolo...\n")

    # Si no existe la libreria, se instala
    subprocess.call([sys.executable, "-m", "pip", "install", 'pyodbc'])
    print("Instalación terminada....")

finally:
    # Luego de instalarse se importa nuevamente la libreria
    import pyodbc

try:
    import dash
except ImportError:
    print("No posee la libreria 'dash'. El programa se encuentra instalandolo...\n")

    # Si no existe la libreria, se instala
    subprocess.call([sys.executable, "-m", "pip", "install", 'dash'])
    print("Instalación terminada....")

finally:
    # Luego de instalarse se importa nuevamente la libreria
    import dash

try:
    import dash_bootstrap_components as dbc
except ImportError:
    print("No posee la libreria 'dash_bootstrap_components'. El programa se encuentra instalandolo...\n")

    # Si no existe la libreria, se instala
    subprocess.call([sys.executable, "-m", "pip", "install", 'dash_bootstrap_components'])
    print("Instalación terminada....")
finally:
    # Luego de instalarse se importa nuevamente la libreria
    import dash_bootstrap_components as dbc

try:

    import pandas as pd

except ImportError:
    print("No posee la libreria 'pandas'. El programa se encuentra instalandolo...\n")

    # Si no existe la libreria, se instala
    subprocess.call([sys.executable, "-m", "pip", "install", 'pandas'])
    print("Instalación terminada....")

finally:
    # Luego de instalarse se importa nuevamente la libreria
    import pandas as pd

try:

    import statsmodels.api as sm

except ImportError:
    print("No posee la libreria 'statsmodels.api'. El programa se encuentra instalandolo...\n")

    # Si no existe la libreria, se instala
    subprocess.call([sys.executable, "-m", "pip", "install", 'statsmodels'])
    print("Instalación terminada....")

finally:
    # Luego de instalarse se importa nuevamente la libreria
    import statsmodels.api as sm

from statsmodels.tsa.seasonal import seasonal_decompose
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import datamanager
import warnings
import itertools
import numpy as np

warnings.filterwarnings("ignore")
warnings.warn("deprecated", DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


PESTANIA_1 = \
"""
Se inició identificando los 10 productos con mayor número de ventas. Estos se recolectaron del Historial de 
Ventas proporcionado por la empresa. Se tomó como parámetro de análisis el producto:  
017027 - Az primera 15x15 Blanco 1 FUJIN..
"""

PESTANIA_2 = \
"""
La serie de tiempo es una secuencia de observaciones de una variable que se mide en puntos sucesivos del tiempo
o en periodos determinados. Donde exploramos los datos en una gráfica de serie de tiempo e identificar su patrón,
ya que las condiciones de un negocio pueden cambiar, haciendo que el patrón de la serie cambie (Anderson et al., 2016).
"""

def generate_table(dataframe, max_rows=10):
    return dbc.Table(
        (
            # Header
            [html.Tr([html.Th(col) for col in dataframe.columns])] +

            # Body
            [html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))]
        ),striped=True, bordered=True, hover=True,
    )



DATA_VENTAS_TOTALES = datamanager.getVentasProducto()

pre_opciones = []

timeSerie_producto = ['012397','017027','014620','015212']

control = 0

for row in DATA_VENTAS_TOTALES.iterrows():
   if(control > 9):
       break
   pre_opciones.append(row[1]['codigo_producto'])

   control += 1



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Creando aplicacion de Dash
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

#Colocando titulo a la pestania
app.title = 'Forecast Ventas'

#Creando Layout de la aplicacion
app.layout = html.Div(children=[
    html.H1(children='Forecast de Ventas Netas'),
    html.H4(children='Pablo Sao & Mercedes Retolaza'),
    html.Br(),
    #html.Div(children=''''''),

    dbc.Tabs([
        dbc.Tab((
            dbc.Card(
                dbc.CardBody(
                    [

                        html.P(PESTANIA_1,className="card-text"),
                        dbc.Row([
                            dbc.Col(
                                dcc.Dropdown(
                                    id='dropdown_productos',
                                    options=[
                                        {'label': i[1]['Producto'], 'value': i[1]['codigo_producto']} for i in DATA_VENTAS_TOTALES.iterrows()
                                    ],
                                    multi=True,
                                    value=pre_opciones,
                                    placeholder='Filtro de Productos.',
                                )
                            )
                        ],
                        ),
                        html.Br(),
                        html.Div(id='tabla-venta-productos'),
                        #dbc.Button("Click here", color="success"),

                        dbc.Row([
                            dbc.Col(
                                # Dropdown para seleccionar productos para ploteo de
                                # serie de tiempo
                                dcc.Dropdown(
                                    id='a_serie_productos',
                                    options=[
                                        {'label': i[1]['Producto'], 'value': i[1]['codigo_producto']} for i in DATA_VENTAS_TOTALES.iterrows()
                                    ],
                                    multi=True,
                                    value=timeSerie_producto,
                                    placeholder='Filtro de Productos.',
                                )
                            )
                        ],
                        ),

                        html.Br(),
                        html.Div(id='serie-venta-productos'),

                    ]
                ),
                className="mt-3",
            )
        ), label="Análisis Exploratorio",disabled=False),
        dbc.Tab(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.P(PESTANIA_2,className="card-text"),
                        dbc.Row([
                            dbc.Col(
                                html.Div(children='''Producto:'''),
                            ),
                            dbc.Col(
                                html.Div(children='''Modelo:'''),
                            ),
                        ]),
                        html.Br(),
                        dbc.Row([
                            dbc.Col(
                                dcc.Dropdown(
                                    id='seleccion_producto',
                                    options=[
                                        {'label': i[1]['Producto'], 'value': i[1]['codigo_producto']} for i in DATA_VENTAS_TOTALES.iterrows()
                                    ],
                                    multi=False,
                                    value='017027',
                                    placeholder='Filtro de Producto',
                                )
                            ),

                            dbc.Col(
                                dcc.Dropdown(
                                    id='seleccion_modelo',
                                    options=[
                                        {'label': 'Adición', 'value': 'additive'},
                                         {'label': 'Multiplicativo', 'value': 'multiplicative'}
                                    ],
                                    multi=False,
                                    value='additive',
                                    placeholder='Filtro Evaluación Modelo',
                                )
                            ),
                            #dbc.Col(
                            #    html.Div(id='serie-tiempo'),
                            #),
                        ],
                        ),
                        html.Br(),

                        #dbc.Row([
                            html.Div(id='serie-tiempo'),
                        #]),



                        #dbc.Button("Click here", color="success"),
                    ]
                ),
                className="mt-3",
            )
            ,label="Forecast",disabled=False),

        # dbc.Tab(
        #     "This tab's content is never seen", label="Forecast", disabled=True
        # ),
    ])

])

@app.callback(
    dash.dependencies.Output('serie-venta-productos', 'children'),
    [dash.dependencies.Input('a_serie_productos', 'value')])
def display_table(dropdown_serieProductos):

    if(len(dropdown_serieProductos) > 0):
        
        cLayout = go.Layout(title='Serie de Tiempo de Productos"',
                       # Same x and first y
                        xaxis_title = 'Fecha',
                        yaxis_title = 'Ventas (Q)'
                       )
        #Lista que contendra las graficas
        trace_producto = []

        for producto in dropdown_serieProductos:
            
            DATOS_PRODUCTOS = datamanager.getAllFechasVentasProducto(producto)

            trace_custom = go.Scatter(x=DATOS_PRODUCTOS.fecha, y=DATOS_PRODUCTOS.ventas, name='Producto: {0}'.format(producto))
            trace_producto.append(trace_custom)

        return (
            dbc.Row([

                dbc.Col(
                    html.Div(
                        dcc.Graph(id='graph', figure={
                            'data': trace_producto,
                            'layout':cLayout
                        })
                    )
                )
            ])
        )

    return (
        dbc.Row([
            dbc.Col(
                html.Div(
                    '''Selecione los Productos para mostrar la grafica'''
                ),
            ),
        ])
    )


@app.callback(
    dash.dependencies.Output('tabla-venta-productos', 'children'),
    [dash.dependencies.Input('dropdown_productos', 'value')])
def display_table(dropdown_value):
    if dropdown_value is None:
        trace1 = go.Bar(x=DATA_VENTAS_TOTALES.Producto, y=DATA_VENTAS_TOTALES.total_ventas, name='Ventas totales')

        return (
            dbc.Row([
                dbc.Col(
                    html.Div(generate_table(DATA_VENTAS_TOTALES))
                ),

                dbc.Col(
                    html.Div(
                        dcc.Graph(id='graph', figure={
                            'data': [trace1],
                            'layout':
                                go.Layout(title='Unidades Totales Vendidas del 2010 al 2014',
                                          barmode='stack')
                        })
                    )
                )
            ])
        )

        #return generate_table(data)

    dff = DATA_VENTAS_TOTALES[DATA_VENTAS_TOTALES.codigo_producto.str.contains('|'.join(dropdown_value))]

    trace1 = go.Bar(x=dff.Producto, y=dff.total_ventas, name='Ventas Totales')

    return (
        dbc.Row([
            dbc.Col(
                html.Div(generate_table(dff))
            ),

            dbc.Col(
                html.Div(
                    dcc.Graph(id='graph', figure={
                        'data': [trace1],
                        'layout':
                            go.Layout(title='Unidades Totales Vendidas del 2010 al 2014',
                                      barmode='stack')
                    })
                )
            )
        ])
    )

@app.callback(
    dash.dependencies.Output('serie-tiempo', 'children'),
    [dash.dependencies.Input('seleccion_producto', 'value'),
     dash.dependencies.Input('seleccion_modelo', 'value') ])
def display_serieTiempo(dropdown_value,dropdown_modelo):


    modelo_evaluar = 'additive'
    tendencia_visible = 'legendonly'

    # Si el dato es de tipo 'NoneType' deja el valor inicial por defecto
    # de lo contrario asigna el modelo seleccionado
    if(isinstance( dropdown_modelo, str )):
        modelo_evaluar = dropdown_modelo
        tendencia_visible = True


    DATA_SERIE = datamanager.getFechasVentasProducto(dropdown_value)


    #min_date = DATA_SERIE.sort_values(by=['fecha']).head(5)['fecha'].tolist()[0]



    # Descomponiendo datos
    data_descomposition = DATA_SERIE.copy(deep=False)
    data_descomposition = data_descomposition.set_index('fecha')

    data_forecast = DATA_SERIE.copy(deep=False)
    data_forecast = data_forecast.set_index('fecha')

    FRECUENCIA = int(DATA_SERIE['ventas'].count()/2)
    
    descomposition = seasonal_decompose(data_descomposition, period=FRECUENCIA, model=modelo_evaluar)

    data_trend = pd.DataFrame(descomposition.trend)
    data_seasonal = pd.DataFrame(descomposition.seasonal)
    data_resid = pd.DataFrame(descomposition.resid)

    '''
            VALIDACION FORECAST
    '''

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    #df = pd.DataFrame(columns=['order', 'param_seasonal', 'AIC'])
    df = pd.DataFrame(columns=['Orden', 'Estacionalidad', 'Ruido (AIC)'])
    #print(data_forecast)
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:

                mod = sm.tsa.statespace.SARIMAX(data_forecast,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                trend='ct',
                                                enforce_stationarity=False,
                                                enforce_invertibility=False,
                                                measurement_error=True)
                results = mod.fit()
                df = df.append({'Orden': param, 'Estacionalidad': param_seasonal, 'Ruido (AIC)': results.aic},
                               ignore_index=True)
                # print('ARIMA{0}x{1}12 - AIC:{2}'.format(param, param_seasonal, results.aic))
            except:
                continue

    df = df.sort_values(by=['Ruido (AIC)']).head(5)
    # print(df)
    # print(df['param'].tolist()[0])

    '''
    Param: (1, 1, 1)
    seasonal: (1, 1, 0, 12)
    '''
    var_order_param =  df['Orden'].tolist()[0]
    var_seasonal_order = df['Estacionalidad'].tolist()[0]

    mod = sm.tsa.statespace.SARIMAX(data_forecast,
                                    order=var_order_param,
                                    seasonal_order= var_seasonal_order,
                                    trend='ct',
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                    measurement_error=True)

    results = mod.fit()



    pred = results.get_prediction(start=pd.to_datetime(pd.to_datetime('2014-01-01')), dynamic=False)
    pred_ci = pred.conf_int()

    #pred_ci['diferencia'] = pred_ci['upper ventas'] - pred_ci['lower ventas']

    pred_ci['diferencia'] = pred_ci.apply(lambda x: (x['upper ventas'] + x['lower ventas'])/2, axis=1)

    '''
        FIN VALIDACION FORECAST
    '''
    #Obteniendo el nombre del producto seleccionado

    nombre_producto = DATA_VENTAS_TOTALES[DATA_VENTAS_TOTALES['codigo_producto'] == dropdown_value]['Producto']

    cLayout = go.Layout(title='Tendencia de Ventas Promedio de "{0}"'.format(nombre_producto.array[0]),
                       # Same x and first y
                        xaxis_title = 'Fecha',
                        yaxis_title = 'Ventas (Q)',
                        height=700
                       )

    trace1 = go.Scatter(x=DATA_SERIE.fecha, y=DATA_SERIE.ventas,name='Ventas Reales (Mensual)')
    trace2 = go.Scatter(x=data_trend.index, y=data_trend.trend,name='Tendencia de Venta',
                        visible='legendonly')
    trace3 = go.Scatter(x=data_seasonal.index, y=data_seasonal.seasonal,
                        name='Tendencia Estacional de Venta',visible=tendencia_visible)
    trace4 = go.Scatter(x=data_resid.index, y=data_resid.resid,name='Residuos',
                        visible='legendonly')

    # print(fechas_prediccion)
    # print(pred2[0].to_list())

    '''
        TRACE FORECAST
    '''

    trace5 = go.Scatter(x=(pred_ci.index).to_list(), y=pred_ci['upper ventas'], name='Margen Positivo',
                        showlegend=True, fill='tozeroy', line=dict(color='rgb(155,155,155)'))

    trace6 = go.Scatter(x=(pred_ci.index).to_list(), y=pred_ci['lower ventas'],name='Margen Negativo', #showlegend=True
                        visible='legendonly',fill='tozeroy',line=dict(color='rgb(155,155,155)'))


    trace7 = go.Scatter(x=(pred_ci.index).to_list(), y=pred_ci['diferencia'], name='Prediccion',
                        showlegend=True,line=dict(color='rgb(255,102,0)'))

    # print(pred_ci)

    return (
        dbc.Row([
            html.H2('Selección de Combinaciones de Parámetros Estacionales'),
        ],align="center",),

        dbc.Row([
            dbc.Col(
                html.Div(
                    '''En los modelos de serie de tiempo es muy común utilizar el método de promedios 
                    móviles, el cual utiliza los valores de los datos más recientes para pronosticar 
                    una serie de tiempo del periodo siguiente. Este es conocido como ARIMA, el cual 
                    utiliza tres parámetros que explican la estacionalidad, tendencia y ruido de los 
                    datos [ ARIMA( p , d , q ) ] (Anderson et al., 2016; Li, 2018). '''
                ),
            ),
            dbc.Col(
                html.Div(generate_table(df)),
            ),
        ]),

        html.Div(
            dcc.Graph(id='graph', figure={
                'data': [trace1,trace2,trace3,trace4,trace5,trace6,trace7],
                'layout': cLayout
            }),
        )
    )

if __name__ == '__main__':
    app.run_server(debug=False,port=5000)