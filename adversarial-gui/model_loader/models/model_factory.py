# Autor: José Luis López Ruiz
# Fecha: 08/05/2024
# Descripción: Este script define el patrón de diseño Factory para crear los modelos.

from .model_interface import ModelInterface
from model_loader.model_utils.model_predictions import ModelPrediction

class AbstractModelFactory():
    def __init__(self) -> None:
        pass

    def create_model(self) -> ModelInterface:
        pass

    def get_name(self) -> str:
        return self.create_model().get_name()
    
    def predict(self, image_path: str) -> tuple[ModelPrediction,list[ModelPrediction]]:
        return self.create_model().predict(image_path)
