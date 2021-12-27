#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings  #evitar warnings
import pandas as pd  #trabajar tablas y estructuras de datos
from pandas import ExcelWriter 
from pandas import ExcelFile
from pandas import datetime
from pandas import DataFrame
import numpy as np #vectores y matrices multidimensiones y operaciones complejas
import matplotlib.pyplot as plt  #para trabajar graficos
from openpyxl import Workbook #en caso de ser necesario instalar la libreria openpyxl y workbook para excel
import statsmodels.api as sm  #explorar modelos estadisticos
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose #descomposición de la serie
from sklearn.metrics import mean_squared_error, mean_absolute_error #biblioteca de aprendizaje automatico
from sklearn.model_selection import train_test_split
from math import sqrt #operaciones matematicas 
from tkinter import filedialog #tkinter trabaja ambiente grafico
import pmdarima as pm #autoarima... se debe de instalar pip install pmdarima

from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta #pip install python-dateutil


# Primer código

# In[3]:


warnings.filterwarnings("ignore")
#variables para cambiar en el modelo 
tiempo_pronosticar=12 #ventana de tiempo a pronosticar
tren='add' #tendencia aditiva o multiplicativa
seas='mul' #estacionalidad multiplicativa
filename =  filedialog.askopenfilename(initialdir = "/",title = "Seleccione el archivo",filetypes = (("Archivos CSV","*.xlsx"),("Todos los Archivos","*.*")))


# In[6]:


df = pd.read_excel(filename, sheet_name='Hoja1') #Modificar en caso de cambiar nombre de hoja
df.info()
df.head(15)


# In[7]:


#descomposición de la serie de tiempo, importante la frecuencia como es por mes, 12 meses
result = seasonal_decompose(df.iloc[:, 1], model=seas, freq=12)
result.plot()
plt.show()


# In[6]:


#Convertir la fecha en un formato pandas 
df.Timestamp = pd.to_datetime (df.iloc[:, 0], format = "%d/%m/%Y") #indexa la fecha, le da formato
df.index = df.Timestamp
df.head(10)

#df.iloc[:,0] tomo la primera columna, es esquivalente a decir df.FECHAS 


# In[7]:


#separación de set de datos, entrenamiento y test

df1=df.iloc[:, 1] #copia de la columna facturación, es equivalente a df.facturacion
#y tiene los valores de facturación
#X tiene los valores de fecha

train=df[0:int(0.9*(len(df)))] #entrenamiento
test=df[int(0.9*(len(df))):]  #test


# In[8]:


#Graficando la data

plt.plot(train.iloc[:, 1], label='Train') #grafica los valores de entrenamiento
plt.plot(test.iloc[:, 1], label='Test') #grafica los valores del test
plt.legend(loc='best')
plt.xlabel('Fecha') #colocar el nombre de las variables
plt.ylabel('Facturacion')
plt.show()


# In[9]:


result = seasonal_decompose(df.iloc[:, 1], model=seas, freq=12)  #descomposición de la serie de tiempo, importante la frecuencia
result.plot()
plt.show()


# In[10]:


#METODO DE HOLT-WINTERS

HW = test.copy()
fit1 = ExponentialSmoothing(np.asarray(train.iloc[:, 1]) ,seasonal_periods=12 ,trend= tren, seasonal=seas).fit()
HW['Holt_Winter'] = fit1.forecast(len(test))

print ('Metricas de Validación HoltWinter')
print('MAPE: ',np.mean(np.abs((test.iloc[:, 1] - HW.Holt_Winter) / test.iloc[:, 1])) * 100)
print('MAE:', mean_absolute_error(test.iloc[:, 1], HW.Holt_Winter))
print('RMSE: ',sqrt(mean_squared_error(test.iloc[:, 1], HW.Holt_Winter)))
print('MSE: ',(mean_squared_error(test.iloc[:, 1], HW.Holt_Winter)))


# In[11]:


#GRÁFICOS HOLTWINTER
plt.plot(train.iloc[:, 1], label='Train')
plt.plot(test.iloc[:, 1], label='Test')
plt.plot(HW.Holt_Winter, label='Holt_Winter')
plt.legend(loc='best')
plt.show()


# In[12]:


#MÉTODO SARIMA AUTOCALIBRACIÓN

model = pm.auto_arima(train.iloc[:, 1], start_p=1, start_q=1,
                         test='adf', #prueba test Fuller
                         max_p=9, max_q=9, m=12, #se pueden cambiar p y q #M es la frecuencia
                         start_P=0, seasonal=True, #componente estacional
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

print(model.summary())
model.plot_diagnostics(figsize=(7,5))
plt.show()


# In[13]:


#PROYECCIÓN SARIMA

SA = test.copy()

model.fit(train.iloc[:, 1])
SA['SARIMA'] = model.predict(n_periods=len(test))

print ('Metricas de Validación SARIMA')
print('MAPE: ',np.mean(np.abs((test.iloc[:, 1] - SA.SARIMA) / test.iloc[:, 1])) * 100)
print('MAE:', mean_absolute_error(test.iloc[:, 1], SA.SARIMA))
print('RMSE: ',sqrt(mean_squared_error(test.iloc[:, 1], SA.SARIMA)))
print('MSE: ',(mean_squared_error(test.iloc[:, 1], SA.SARIMA)))

#METODO DE HOLT-WINTERS


# In[14]:


#GRÁFICOS SARIMA
plt.plot(train.iloc[:, 1], label='Train')
plt.plot(test.iloc[:, 1], label='Test')
plt.plot(SA.SARIMA, label='Holt_Winter')
plt.legend(loc='best')
plt.show()


# In[15]:


#Crear proyecciones Holtwinter

fit_1 = ExponentialSmoothing((df1) ,seasonal_periods=12 ,trend=tren, seasonal=seas).fit()
HW_forcast = fit_1.forecast(tiempo_pronosticar)
print('Proyecciones HoltWinter')
HW_forcast.head(100)


# In[16]:


#pronosticar Autosarima
models = pm.auto_arima(df.iloc[:, 1], start_p=1, start_q=1,
                         test='adf', #prueba test Fuller
                         max_p=9, max_q=9, m=12, #se pueden cambiar p y q #M es la frecuencia
                         start_P=0, seasonal=True, #componente estacional
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)


fecha=df.index[-1]+ relativedelta(months=1)
AS_forcast2 = models.predict(n_periods=tiempo_pronosticar)
indexar_fecha = pd.date_range(fecha, periods = tiempo_pronosticar, freq='MS')
AS_forcast = pd.Series(AS_forcast2, index=indexar_fecha)
print('Proyecciones SARIMA')
AS_forcast.head(100)


# In[17]:


df1.plot(kind="line",figsize=(10,5),legend=True)
HW_forcast.plot(kind="line",figsize=(10,5),color='orange',legend=True,label='Holt-Winters')
AS_forcast.plot(kind="line",figsize=(10,5),color='red',legend=True,label='Sarima')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




