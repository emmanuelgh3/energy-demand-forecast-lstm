# ERCOT Far West Electricity Demand Forecasting (LSTM)

**Predicción horaria de demanda energética en la zona Far West del estado de Texas usando redes neuronales recurrentes (RNN)**  

![Predicciones vs Demanda Real](reports/figures/predictions_2025.png) *(Actualiza con tu gráfico principal)*

## Descripción del Proyecto y Objetivos
Este proyecto se centra en predecir la demanda energética horaria (en MWh) de la región Far West de Texas (FWES) de ERCOT utilizando redes neuronales recurrentes (RNN). En especifico, se hará uso de LSTM (Long Short-Term Memory). El objetivo es optimizar la generación de energía para evitar sobrecostos y mejorar la eficiencia en la gestión de la red eléctrica, dado el alto grado de penetración de energías renovables y la volatilidad extrema de la demanda y generación. Usando datos históricos horarios del 2019 al 2024 y 

## ¿Por qué Far West?
La región de Far West Texas es una de las más relevantes en términos de generación de energía eólica (energías renovables) y presenta una alta volatilidad en la demanda y generación. Por ello, la predicción precisa de la demanda es importante para evitar cuellos de botella en la transmisión y optimizar la generación de energía.

## Estructura del Proyecto
*   **`LSTM.ipynb`**: El código principal del modelo de predicción utilizando LSTM.
*   **`requirements.txt`**: Lista de dependencias necesarias para ejecutar el proyecto.
*   **`README.md`**: Este archivo, que explica cómo ejecutar el proyecto y qué hace.
*   **`.gitignore`**: Archivos que no deben ser incluidos en el repositorio.

