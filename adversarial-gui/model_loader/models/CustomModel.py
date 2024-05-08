# Autor: José Luis López Ruiz
# Fecha: 08/05/2024
# Descripción:  Este script define la clase ModelCustomModel, la cual implementa la interfaz CustomModelInterface y representa un modelo de red neuronal el cual puede ser instanciado a partir de la ruta de un modelo con formato .h5.


from model_loader.model_utils.model_predictions import ModelPrediction
from model_loader.model_utils.model_exceptions import ModelNotLoadedException
from .model_interface import ModelInterface, CustomModelInterface

import tensorflow as t
from keras.preprocessing import image
import numpy as n
from typing import List

class ModelCustomModel(CustomModelInterface):
    def __init__(self, model_path: str) -> None:
        try:
            self.model = t.keras.models.load_model(model_path)
            self.model_name = "Custom: " + model_path.split("/")[-1].split(".")[0]
        except Exception:
            raise ModelNotLoadedException()

    def get_name(self) -> str:
        return self.model_name

    # Por implementar
    def predict(self, image_path: str) -> tuple[ModelPrediction,List[ModelPrediction]]:
        pass

    
