# Autor: José Luis López Ruiz
# Fecha: 04/05/2024
# Descripción: Este script contiene la clase ModelLoader, la cual se encarga de cargar un modelo de red neuronal.

from model_loader.model_utils.model_exceptions import InvadidModelName, ModelNotLoadedException
from model_loader.model_utils.model_predictions import ModelPrediction

from model_loader.models.ResNet50 import ModelResNet50
from model_loader.models.SignalModel import ModelSignalModel

import os


class ModelLoader(): 

    def __init__(self): 
        self.models = {}
        self.model = None

        # Cargar modelo por defecto
        self.switch_model("ResNet50")

    def init_model(self, str_model: str): 
        print(f"Cargando modelo: {str_model}")
        if not str_model:
            raise InvadidModelName(model_name=str_model)
        
        if str_model == "ResNet50": 
            self.model = ModelResNet50()
            print(f"Modelo cargado: {self.model.get_name()}")

        elif str_model == "SignalModel":
            self.model = ModelSignalModel()
            print(f"Modelo cargado: {self.model.get_name()}")

        else:
            raise InvadidModelName(model_name=str_model)
        
        self.models.update({self.model.get_name() : self.model})

    
    def switch_model(self, str_model: str):
        if self.model is None:
            self.init_model(str_model)
        else:
            try:
                self.model = self.models[str_model]
            except KeyError:
                self.init_model(str_model)


    def predict(self, image: str) -> tuple[ModelPrediction,list[ModelPrediction]]:
        if self.model is None:
            raise ModelNotLoadedException()
        else:
            return self.model.predict(image)
        