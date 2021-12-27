#importo librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Regresion.csv')
#profe no tenía el dataset en columnas, entonces lo dividí
data = dataset["Fecha;Generacion de ingresos de no intermediacion financiera"].str.split(";", n=1, expand = True)

#convertir en formato fecha y número para meterla en la serie de tiempo
import datetime as dt
data[0] = pd.to_datetime(data[0])
data[0] = data[0].map(dt.datetime.toordinal)

X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

#Profe acá divido en entrenamiento y test
from sklearn.model_selection import train_test_split
#1 de cada 3 es la fase de testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) 

#crear modelo de regresión lineal
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

#predecir el conjunto de test

y_pred = regression.predict(X_test)

#visualizar los resultados de entrenamiento
plt.scatter(X_train,y_train, color="red")
plt.plot(X_train,regression.predict(X_train), color="blue")
plt.title("Años vs Generación de ingresos(conjunto entrenamiento)")
plt.xlabel("Fechas")
plt.ylabel("% generación de ingresos")
plt.show()
#visualizar los resultados de test
plt.scatter(X_test,y_test, color="green")
plt.plot(X_train,regression.predict(X_train), color="blue")
plt.title("Años vs Generación de ingresos(conjunto de test)")
plt.xlabel("Fechas")
plt.ylabel("% generación de ingresos")
plt.show()

#Predicciones requeridas de años (dicimebre 2018 y hasta Octubre 2019)

Fechasnuevas = pd.read_csv('Fechas_pred.csv')

import datetime as dt
Fechasnuevas["Fechas"] = pd.to_datetime(Fechasnuevas["Fechas"])
Fechasnuevas["Fechas"] = Fechasnuevas["Fechas"].map(dt.datetime.toordinal)

X_nueva = Fechasnuevas.iloc[:, :].values

#Proyección con fechas '01/12/2018', '01/01/2019','01/02/2019','01/03/2019','01/03/2019','01/04/2019','01/05/2019','01/06/2019','01/07/2019','01/08/2019'

Proyeccion = regression.predict(X_nueva)






