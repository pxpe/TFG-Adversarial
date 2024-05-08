# Autor: José Luis López Ruiz
# Fecha: 04/05/2024
# Descripción: Este script contiene la clase ModelLoader, la cual se encarga de cargar un modelo de red neuronal.

from model_loader.model_utils.model_exceptions import InvadidModelName, ModelNotLoadedException
from model_loader.model_utils.model_predictions import ModelPrediction


class ModelLoader(): 
    def __init__(self): 
        self.models = []
        self.model = None

        # Cargar modelo por defecto
        self.init_model("ResNet50")

    def init_model(self, str_model: str): 
        print(f"Cargando modelo: {str_model}")
        if str_model == "ResNet50": 
            from model_loader.models.ResNet50 import ModelResNet50
            self.model = ModelResNet50()
            self.models.append(self.model)
            print(f"Modelo cargado: {self.model.get_name()}")
        else:
            raise InvadidModelName(model_name=str_model)
        

    def predict(self, image: str) -> list[ModelPrediction]:
        if self.model is None:
            raise ModelNotLoadedException()
        else:
            return self.model.predict(image)
        