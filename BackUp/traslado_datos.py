import subprocess
import sys
import datetime

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


def add_record(codigo_sucursal,fecha,sucursal,codigo_producto,Producto,total_vendido):

    query = "insert into ventas (codigo_sucursal,fecha,sucursal,codigo_producto,Producto,total_vendido)"\
            " values({0},'{1}','{2}','{3}','{4}',{5})".format(codigo_sucursal,fecha,sucursal,codigo_producto,Producto,total_vendido)
    try:

        con = pyodbc.connect('Driver={SQL Server};'
                              'Server=localhost;'
                              'Database=ventas;'
                              'UID=sa;'
                              'PWD=123456;'
                              'Trusted_Connection=No;')


        cur = con.cursor()

        cur.execute(query)
        print("Producto ingresado: {0}...".format(Producto))
        con.commit()

        cur.close()
        con.close()

    except Exception as e:
        print(str(e))
        exit()

try:

    print("\t\tIniciando Programa...\n")
    conn = pyodbc.connect('Driver={SQL Server};'
                          'Server=localhost;'
                          'Database=MilleniumIM;'
                          'UID=sa;'
                          'PWD=123456;'
                          'Trusted_Connection=No;')

    cursor = conn.cursor()
    print("\t\tObteniendo datos...\n")
    cursor.execute("EXECUTE [dbo].[Stp_UdPvVentas] '2000-01-01 00:00:00.000','2018-12-31 00:00:00.000'")

    rows = cursor.fetchall()

    control = 0

    for row in rows:
        fecha = row.fecha
        fecha = fecha.replace(day=1)
        print("Agregando producto: {0}...".format(row.Producto))
        add_record(row.codigo_sucursal, fecha, row.sucursal, row.codigo_producto, row.Producto, row.total_vendido)

    cursor.close()
    conn.close()

except Exception as e:
    print(str(e))