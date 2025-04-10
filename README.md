# ERCOT Far West Electricity Demand Forecasting (LSTM)

**Predicción horaria de demanda energética en la zona Far West del estado de Texas usando redes neuronales recurrentes (RNN)**  

![Predicciones vs Demanda Real](images/overview.png)

## Descripción del Proyecto y Objetivos
Este proyecto se centra en predecir la demanda energética horaria (en MWh) de la región Far West de Texas (FWES) de ERCOT utilizando redes neuronales recurrentes (RNN). En especifico, se hará uso de LSTM (Long Short-Term Memory). El objetivo es optimizar la generación de energía para evitar sobrecostos y mejorar la eficiencia en la gestión de la red eléctrica, dado el alto grado de penetración de energías renovables y la volatilidad extrema de la demanda y generación. Usando datos históricos horarios del 2019 al 2024 y 

## ¿Por qué Far West?
La región de Far West Texas es una de las más relevantes en términos de generación de energía eólica (energías renovables) y presenta una alta volatilidad en la demanda y generación. Por ello, la predicción precisa de la demanda es importante para evitar cuellos de botella en la transmisión y optimizar la generación de energía.

## Estructura del Proyecto
*   **`LSTM.ipynb`**: El código principal del modelo de predicción utilizando LSTM.
*   **`requirements.txt`**: Lista de dependencias necesarias para ejecutar el proyecto.
*   **`README.md`**: Este archivo, que explica cómo ejecutar el proyecto y qué hace.
*   **`.gitignore`**: Archivos que no deben ser incluidos en el repositorio.

## Descripción de los Datos

Los datos utilizados en este proyecto provienen de la **región Far West de Texas (FWES)** dentro de **ERCOT (Electric Reliability Council of Texas)**, específicamente de la demanda de energía eléctrica horaria. La región de FWES es conocida por su alta penetración de energías renovables, principalmente eólica, y presenta una volatilidad extrema tanto en la demanda como en la generación de electricidad.

### Fuentes de Datos

Los datos se obtienen de la **U.S. Energy Information Administration (EIA)** y están disponibles en formato CSV. Estos datos incluyen:

- **Fecha y hora**: El tiempo de cada registro de demanda.
- **Demanda de energía**: La demanda horaria de energía en **megavatios hora (MWh)**.
- **Características de la generación**: Información sobre la cantidad de energía generada por diferentes fuentes (eólica, solar, térmica, etc.), aunque estos datos pueden no estar directamente incluidos en todos los conjuntos de datos.

### Preprocesamiento

Los datos han sido preprocesados de la siguiente manera:
- Eliminación de columnas no relevantes para el analisis. Solo se trabajó con las columnas de **Fecha y hora** y **Demanda de energía**.
- Normalización de los valores de la demanda utilizando **MinMaxScaler**.
- Creación de secuencias temporales para la entrada del modelo LSTM.
- División en conjuntos de entrenamiento y prueba para evaluar el rendimiento del modelo.

Estos datos serán utilizados para entrenar un modelo LSTM que prediga la demanda futura de energía en función de patrones históricos.
