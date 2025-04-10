#!/usr/bin/env python
# coding: utf-8

# # Importacion de librerias

# In[62]:


import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
#Visualizacion de graficas
plt.style.use('seaborn-v0_8-darkgrid')


# # Importacion de la base de datos

# Creamos el df con la base de datos .csv.

# In[5]:


ercot_fwes = pd.read_csv('ercot_fwes_complete_2019-2025.csv')


# # Analisis exploratorio y limpieza de la base de datos

# Desplegamos las primeras y ultimas filas de la base de datos.

# In[6]:


ercot_fwes.head()


# In[7]:


ercot_fwes.tail()


# Observamos la estructura general del dataframe.

# In[8]:


print("\nInformación del DataFrame:")
ercot_fwes.info()


# Realizamos un analisis exploratorio de los datos.

# In[9]:


ercot_fwes.describe()


# Analizamos si existen outliers en la base de datos mediante un boxplot.

# In[10]:


sns.boxplot(x=ercot_fwes['value'])
plt.title("Distribución de la demanda (MWh)")
plt.show()


# Convertimos la variable timestamp en una variable a tipo datetime y verificamos la ordenación de los datos.

# In[11]:


ercot_fwes['timestamp'] = pd.to_datetime(ercot_fwes['timestamp'])
ercot_fwes = ercot_fwes.sort_values('timestamp').reset_index(drop=True)
ercot_fwes.index.is_monotonic_increasing


# Verificamos que no existan valores faltantes y datos duplicados.

# In[12]:


print(ercot_fwes.isna().sum())


# In[13]:


print(f"Duplicados: {ercot_fwes.duplicated().sum()} filas")


# Para aplicar el modelo LSTM que trabaja con datos temporales tenemos que asegurarnos que exista una frecuencia horaria constante en nuestra base de datos, en este caso de una hora entre cada registro.

# In[14]:


ercot_fwes['diff'] = ercot_fwes['timestamp'].diff()
print(ercot_fwes['diff'].value_counts().head())


# Existen 6 registros con un salto de 2 horas. Para solucionar el problema primero haremos un reindex para establecer la columna 'timestamp' como el indice de la base de datos y realizamos una interpolación lineal para rellenar los espacios faltantes (los espacios en donde habia un salto de 2 horas).
# 
# Por practicidad renombraremos la columna 'value' por 'demand_MWh'.

# In[15]:


#establecemos el timestamp como índice
ercot_fwes.set_index('timestamp', inplace=True)


# In[16]:


date_range_full = pd.date_range(
    start=ercot_fwes.index.min(),
    end=ercot_fwes.index.max(),
    freq='1h'
)
ercot_fwes = ercot_fwes.reindex(date_range_full)
ercot_fwes.index.name = 'timestamp'


# In[17]:


ercot_fwes['demand_MWh'] = ercot_fwes['value'].interpolate(method='linear')


# Confirmamos que no existan valores faltantes.

# In[18]:


print("Valores faltantes restantes:", ercot_fwes['demand_MWh'].isna().sum())


# Eliminamos la columna 'diff' y la columna 'value' y solo conservamos 'demand_MWh'.

# In[19]:


ercot_fwes.drop(columns=['value', 'diff'], inplace=True)


# Imprimimos la nueva dimensión de la base de datos.

# In[20]:


ercot_fwes.shape


# Verificamos que todos los datos esten espaciados en una hora.

# In[21]:


print(ercot_fwes.index.to_series().diff().value_counts())


# Podemos graficar la demanda de cada año completo (2020-2024) para dar una idea de los datos con los que estaremos tratando.

# In[22]:


fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15, 20), sharex=False)

years = [2020, 2021, 2022, 2023, 2024]

#subplots para cada año
for i, year in enumerate(years):
    yearly_data = ercot_fwes.loc[str(year)]
    
    axes[i].plot(yearly_data.index, yearly_data['demand_MWh'], color='tab:blue', linewidth=0.7)
    axes[i].set_title(f'Demanda eléctrica en el año {year}', fontsize=12)
    axes[i].set_ylabel('MWh')
    axes[i].grid(True)

plt.tight_layout()
plt.xlabel("Fecha")
plt.suptitle("Demanda eléctrica horaria en la región Far West de Texas por año", fontsize=16, y=1.02)
plt.show()


# # Aplicacion del modelo LSTM

# ## Escalamiento de los datos

# Comenzamos aplicando un MinMaxScaler a los datos de la demanda de energía.

# In[23]:


#Aplicamos MinMaxScaler para escalar la demanda entre 0 y 1
scaler = MinMaxScaler(feature_range=(0, 1))
demand_scaled = scaler.fit_transform(ercot_fwes[['demand_MWh']])

#Añadimos la columna de la demanda escalada al DataFrame
ercot_fwes['demand_scaled'] = demand_scaled

#Imprimimos el dataframe para confirmar el escalamiento
ercot_fwes[['demand_MWh', 'demand_scaled']].head()


# In[24]:


#Grafica de la totalidad de los datos escalados
plt.figure(figsize=(12, 4))
plt.plot(ercot_fwes.index, ercot_fwes['demand_scaled'], label='Demanda escalada')
plt.title('Demanda Horaria Normalizada')
plt.legend()
plt.show()


# ## Creación de secuencias

# Primero definimos la función para crear las secuencias de entrada y etiquetas de salida.

# In[25]:


def create_sequences(data, time_step):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])  #Secuencia de las últimas time_step horas
        y.append(data[i, 0])  #La demanda de la hora objetivo
    return np.array(X), np.array(y)


# Usaremos un time_step de 168 horas (1 semana), para poder predecir la demanda en algun dia y hora arbitrarios.

# In[26]:


time_step = 168
X, y = create_sequences(ercot_fwes[['demand_scaled']].values, time_step) #secuencias con la variable escalada
X = X.reshape(X.shape[0], X.shape[1], 1)


# Verificamos las dimensiones de X, y.

# In[27]:


print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}")


# ## Implementación del modelo LSTM

# Primero dividimos los conjuntos de datos en un conjunto de prueba y uno de entrenamiento.

# In[28]:


#80% entrenamiento/20% prueba (sin mezclar)
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

print(f"Train shapes: {X_train.shape}, {y_train.shape}")
print(f"Test shapes: {X_test.shape}, {y_test.shape}")


# Comenzamos a construir el modelo LSTM con el optimizador ADAM.

# In[30]:


model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
#Se utilizó el optimizador adam con su learning rate predeterminado de 0.001
model.summary()


# Ahora con el modelo LSTM definido realizamos el entrenamiento con los datos X_train y y_train. Implementamos EarlyStopping para detener el entrenamiento basandonos en el error de validación cuando el modelo deje de mejorar. También agregamos ModelCheckpoint para guardar el modelo con el mejor rendimiento durante el entrenamiento para no perder el progreso en caso de existir un sobreajuste.

# In[41]:


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_model.h5',        
    monitor='val_loss',     
    save_best_only=True,    
    mode='min',
    verbose=1 
)


# In[42]:


history = model.fit(
    X_train, y_train,
    epochs=100,  
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint],
    verbose=1
)


# # Evaluación del modelo

# Usaremos nuestro modelo para predecir la demanda electrica en el primer trimestre del año 2025 con los datos obtenidos de EIA.

# In[144]:


ercot_fwes_25 = pd.read_csv('ercot_fwes_010125-033125.csv')


# In[145]:


ercot_fwes_25['timestamp'] = pd.to_datetime(ercot_fwes_25['timestamp'])
ercot_fwes_25.set_index('timestamp', inplace=True)
ercot_fwes_25.head()


# Realizamos la normalizacion de los datos aplicando el mismo scaler que a los datos utilizados para en entrenamiento.

# In[146]:


#Normalizar los valores con el mismo scaler del entrenamiento
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(ercot_fwes[['demand_MWh']])
joblib.dump(scaler, 'min_max_scaler.pkl')


# In[147]:


ercot_fwes_25 = ercot_fwes_25.rename(columns={'value': 'demand_MWh'})
ercot_fwes_25['demand_scaled'] = scaler.transform(ercot_fwes_25[['demand_MWh']])


# In[148]:


#Valor máximo de demand_scaled
max_scaled = ercot_fwes_25['demand_scaled'].max()
print(f"Valor máximo de demand_scaled: {max_scaled}")

#Verificacion de valores fuera de 0-1
out_of_bounds = ercot_fwes_25[(ercot_fwes_25['demand_scaled'] < 0) | (ercot_fwes_25['demand_scaled'] > 1)]
print(f"\nNumero de valores fuera del rango [0, 1]: {len(out_of_bounds)}")


# In[149]:


#Grafica de la totalidad de los datos escalados
plt.figure(figsize=(12, 4))
plt.plot(ercot_fwes_25.index, ercot_fwes_25['demand_scaled'], label='Demanda escalada')
plt.title('Demanda Horaria Normalizada')
plt.legend()
plt.show()


# Ahora creamos las secuencias temporales para predecir la demanda en el primer trimestre del 2025 utilizando el mejor modelo obtenido durante el entrenamiento.

# In[150]:


# Cargar el modelo entrenado
model = load_model('best_model.h5')


# In[151]:


#Crear secuencias temporales 
def create_sequences(data, window_size=168):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
    return np.array(X)

X_2025 = create_sequences(ercot_fwes_25['demand_scaled'].values)
X_2025 = X_2025.reshape(X_2025.shape[0], X_2025.shape[1], 1)  # Shape: (muestras, 168, 1)

#Predecir todas las secuencias
y_pred_scaled = model.predict(X_2025)
y_pred = scaler.inverse_transform(y_pred_scaled).flatten()  # Revertir escalamiento

#Alinear con timestamps reales (descartando primeras 168 horas/primera semana)
timestamps = ercot_fwes_25.index[168:]
real_values = ercot_fwes_25['demand_MWh'].values[168:]


# In[152]:


plt.figure(figsize=(12, 6))
plt.plot(ercot_fwes_25.index[time_step:], ercot_fwes_25['demand_MWh'][time_step:], label='Demanda Real', color='blue')
plt.plot(predicted_df.index, predicted_df['predicted_demand'], label='Demanda Predicha', color='red', linestyle='--')
plt.title('Comparacion Entre Demanda Real y Demanda Predicha para el Primer Trimestre de 2025')
plt.xlabel('Fecha')
plt.ylabel('Demanda (MWh)')
plt.legend()
plt.show()


# In[158]:


# Calcular R-squared mensual
r_squared_monthly = []

# Agrupar por mes y calcular el R-squared para cada mes
for month, group in predicted_df.groupby(predicted_df.index.month):
    # Filtrar los valores reales para el mismo mes que las predicciones
    real_for_month = real_values[predicted_df.index.month == month]
    pred_for_month = group['predicted_demand']
    
    r_squared = r2_score(real_for_month, pred_for_month)
    r_squared_monthly.append((month, r_squared))

    print(f'R-squared para el mes {month}: {r_squared}')

# Gráfico interactivo con Plotly
fig = go.Figure()

# Añadir la línea de la demanda real
fig.add_trace(go.Scatter(
    x=ercot_fwes_25.index[168:],
    y=real_values,
    mode='lines',
    name='Demanda Real',
    line=dict(color='blue')
))

# Añadir la línea de la demanda predicha
fig.add_trace(go.Scatter(
    x=predicted_df.index,
    y=predicted_df['predicted_demand'],
    mode='lines',
    name='Demanda Predicha',
    line=dict(color='red', dash='dash')
))

fig.update_layout(
    title='Comparacion Entre Demanda Real y Demanda Predicha para el Primer Trimestre de 2025',
    xaxis_title='Fecha',
    yaxis_title='Demanda (MWh)',
    template='seaborn',
    legend=dict(x=0, y=1, traceorder='normal', orientation='h')
)

fig.show()

