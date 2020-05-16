"""
Manejo de las consultas a la base de datos
:author: Pablo Sao
:date: 1 de mayo de 2020
"""
# Importamos libreria necesaria para ejecutar programa
import subprocess
import sys

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

    import pyodbc

except ImportError:
    print("No posee la libreria 'pymssql'. El programa se encuentra instalandolo...\n")

    # Si no existe la libreria, se instala
    subprocess.call([sys.executable, "-m", "pip", "install", 'pyodbc'])
    print("Instalación terminada....")

finally:
    # Luego de instalarse se importa nuevamente la libreria
    import pyodbc


def getConnection():
    conn = pyodbc.connect('Driver={SQL Server};'
                          'Server=localhost;'
                          'Database=ventas;'
                          'Trusted_Connection=Yes;')

    return conn

def getVentasProducto():
    """
    Obtiene las ventas totales en toda la hisotria de la empresa
    :return: dataframe de pandas con el codigo de producto, nombre del producto y el total de ventas
    """
    Qventas_totales = """
    SELECT 
         LTRIM(RTRIM(codigo_producto)) AS codigo_producto
        ,Producto
        ,sum(total_vendido) AS total_ventas
    FROM
        ventas
    WHERE
        fecha BETWEEN '2004-01-01 00:00:00.000' AND '2014-12-31 00:00:00.000'
    AND LEFT(codigo_producto, 2) <> 'DF'
    GROUP BY
        codigo_producto, Producto
    ORDER BY total_ventas DESC
    """

    data = pd.DataFrame()

    try:
        data = pd.read_sql(Qventas_totales, getConnection())
    except Exception as e:
        print(str(e))

    return data

def getFechasVentasProducto(codigo_producto):
    """
    Metodo para obtener las ventas promedios mensuales del poducto deseado
    :param codigo_producto:
    :return:
    """
    Qfechaventas_producto = """
        SELECT
             fecha
            ,avg(total_vendido) AS ventas
        FROM 
            ventas
        WHERE
            fecha BETWEEN '2010-01-01 00:00:00.000' AND '2014-12-01 00:00:00.000'
        AND codigo_producto = '{0}'
        GROUP BY fecha
        ORDER BY
            fecha ASC
        """.format(codigo_producto)

    data = pd.DataFrame()

    try:
        data = pd.read_sql(Qfechaventas_producto, getConnection())
    except Exception as e:
        print(str(e))

    return data