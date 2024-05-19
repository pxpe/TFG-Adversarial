# Autor: José Luis López Ruiz
# Fecha: 08/05/2024
# Descripción: Este script define la clase ModelPrediction la cual define una Prediccion con dos atributos de clase.
# - predicted_class_reliability: Fiabilidad de la predicción en porcentaje  
# - predicted_class: Clase predicha por el 
# Además, define la función generate_graph que genera un gráfico de barras con las ultimas cuatro predicciones del modelo.


from typing import List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

class ModelPrediction():
    """
        Clase que define una Predicción con dos atributos de clase.
        - predicted_class: Clase predicha por el modelo.
        - predicted_class_reliability: Fiabilidad de la predicción en porcentaje
    """
    def __init__(self, predicted_class : str, predicted_class_reliability : str):
        self.predicted_class = predicted_class
        self.predicted_class_reliability = predicted_class_reliability
    
    def __str__(self):
        return f"({self.predicted_class}, {self.predicted_class_reliability}%)"
    

def generate_prediction_graph(model_predictions : List[ModelPrediction]) -> tuple[Figure, Axes]:

    """
        Función que genera un gráfico de barras con las ultimas cuatro predicciones del modelo.
    """

    model_predictions = model_predictions[:4]

    fig, ax = plt.subplots(figsize=(3, 2))
    # ponemos las etiquetas de las barras con tamaño 5
    ax.tick_params(labelsize=5)


    labels = [prediction.predicted_class for prediction in model_predictions]
    probabilities = [round(float(prediction.predicted_class_reliability),2)  for prediction in model_predictions]
    bar_colors = ['tab:red', 'tab:blue', 'tab:olive', 'tab:green']

    ax.bar(x=[i for i in range(0,4)],height=probabilities,tick_label=labels, color=bar_colors)
    ax.set_title('Predicciones del modelo en %')

    return (fig, ax)

