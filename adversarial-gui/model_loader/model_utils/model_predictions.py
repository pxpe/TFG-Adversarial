# Autor: José Luis López Ruiz
# Fecha: 08/05/2024
# Descripción: Este script define la clase ModelPrediction la cual define una Prediccion con dos atributos de clase.
#  - predicted_class: Clase predicha por el modelo.
#  - predicted_class_reliability: Fiabilidad de la predicción en porcentaje


class ModelPrediction():
    def __init__(self, predicted_class : str, predicted_class_reliability : str):
        self.predicted_class = predicted_class
        self.predicted_class_reliability = predicted_class_reliability
    
    def __str__(self):
        return f"({self.predicted_class}, {self.predicted_class_reliability}%)"

