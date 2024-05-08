# Autor: José Luis López Ruiz
# Fecha: 04/05/2024
# Descripción: Este script contiene la clase ModelLoader, la cual se encarga de cargar un modelo de red neuronal.

class ModelLoader(): 
    def __init__(self): 
        self.model = None

    def init_model(self, str_model: str): 
        self.model = str_model
        print(f"Modelo cargado: {self.model}")

    def predict(self, image: str): 
        print(f"Predicción realizada con el modelo {self.model}")
        