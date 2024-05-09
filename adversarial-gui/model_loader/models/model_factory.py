# Autor: José Luis López Ruiz
# Fecha: 08/05/2024
# Descripción: Este script define el patrón de diseño Factory para crear los modelos.

from .model_interface import ModelInterface
from model_loader.model_utils.model_predictions import ModelPrediction

class AbstractModelFactory():
    """
        Clase abstracta que define el patrón de diseño Factory para crear los modelos.
    """

    def __init__(self) -> None:
        pass

    def create_model(self) -> ModelInterface:
        """
            Crea un modelo.
        """
        pass

    def get_name(self) -> str:
        """
            Devuelve el nombre del modelo.
        """
        return self.create_model().get_name()
    
    def predict(self, image_path: str) -> tuple[ModelPrediction,list[ModelPrediction]]:
        """
            Realiza una predicción sobre una imagen.
        """
        return self.create_model().predict(image_path)
