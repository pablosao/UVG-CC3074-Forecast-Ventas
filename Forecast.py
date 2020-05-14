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


PESTANIA_1 = \
"""
Iniciaremos identificando los 10 productos con mayor numero de ventas en la historia de ventas
de la empresa, seleccionando un producto para realizar el análisis. Seleccionando el producto
017027 - Az primera 15x15 Blanco 1 FUJIN.
"""

PESTANIA_2 = \
"""
La serie de tiempo es una secuencia de observaciones de una variable que se mide en puntos sucesivos del tiempo
o en periodos determinados. Donde exploramos los datos en una grafica de serie de tiempo e identificar su patron,
ya que las condiciones de un negocio pueden cambiar, haciendo que el patron de la serie cambie ().
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
    html.H1(children='Análisis de Forecast de Ventas'),

    html.Div(children='''
        
    '''),

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

                        html.Div(id='tabla-venta-productos'),
                        #dbc.Button("Click here", color="success"),
                    ]
                ),
                className="mt-3",
            )
        ), label="Explorando Datos",disabled=False),
        dbc.Tab(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.P(PESTANIA_2,className="card-text"),
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

                            # dbc.Col(
                            #     dcc.Dropdown(
                            #         id='seleccion_modelo',
                            #         options=[
                            #             {'label': 'Adición', 'value': 'additive'},
                            #             {'label': 'Multiplicativo', 'value': 'multiplicative'}
                            #         ],
                            #         multi=False,
                            #         #value='017027',
                            #         placeholder='Filtro Evaluación Modelo',
                            #     )
                            # ),
                            dbc.Col(
                                html.Div(id='serie-tiempo'),
                            ),
                        ],
                        ),




                        #dbc.Button("Click here", color="success"),
                    ]
                ),
                className="mt-3",
            )
            ,label="Análisis Exploratorio",disabled=False),

        dbc.Tab(
            "This tab's content is never seen", label="Forecast", disabled=True
        ),
    ])

])

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
                                go.Layout(title='Unidades Totales vendidas',
                                          barmode='stack')
                        })
                    )
                )
            ])
        )

        #return generate_table(data)

    dff = DATA_VENTAS_TOTALES[DATA_VENTAS_TOTALES.codigo_producto.str.contains('|'.join(dropdown_value))]

    trace1 = go.Bar(x=dff.Producto, y=dff.total_ventas, name='Ventas totales')

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
                            go.Layout(title='Unidades Totales vendidas',
                                      barmode='stack')
                    })
                )
            )
        ])
    )

@app.callback(
    dash.dependencies.Output('serie-tiempo', 'children'),
    [dash.dependencies.Input('seleccion_producto', 'value')])
def display_serieTiempo(dropdown_value):


    DATA_SERIE = datamanager.getFechasVentasProducto(dropdown_value)

    # Descomponiendo datos
    data_descomposition = DATA_SERIE.copy(deep=False)
    data_descomposition = data_descomposition.set_index('fecha')

    FRECUENCIA = int(DATA_SERIE['ventas'].count()/2)

    descomposition = seasonal_decompose(data_descomposition, period=FRECUENCIA, model='multiplicative')

    data_trend = pd.DataFrame(descomposition.trend)
    data_seasonal = pd.DataFrame(descomposition.seasonal)
    data_resid = pd.DataFrame(descomposition.resid)

    #Obteniendo el nombre del producto seleccionado

    nombre_producto = DATA_VENTAS_TOTALES[DATA_VENTAS_TOTALES['codigo_producto'] == dropdown_value]['Producto']

    cLayout = go.Layout(title='Tendencia de Ventas Promedio de "{0}"'.format(nombre_producto.array[0]),
                       # Same x and first y
                        xaxis_title = 'Fecha',
                        yaxis_title = 'Ventas (Q)',
                        height=700
                       )

    trace1 = go.Scatter(x=DATA_SERIE.fecha, y=DATA_SERIE.ventas,name='Observado')
    trace2 = go.Scatter(x=data_trend.index, y=data_trend.trend,name='Tendencia',visible='legendonly')
    trace3 = go.Scatter(x=data_seasonal.index, y=data_seasonal.seasonal,name='Tendencia Estacional',visible='legendonly')
    trace4 = go.Scatter(x=data_resid.index, y=data_resid.resid,name='Residuos',visible='legendonly')


    return (
        html.Div(
            dcc.Graph(id='graph', figure={
                'data': [trace1,trace2,trace3,trace4],
                'layout': cLayout
            }),
        )
    )

if __name__ == '__main__':
    app.run_server(debug=True,port=5000)