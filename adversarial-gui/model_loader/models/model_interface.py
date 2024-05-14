# Autor: José Luis López Ruiz
# Fecha: 08/05/2024
# Descripción: Este script define la Interfaz común para todos los modelos del Model Loader.

from model_loader.model_utils.model_predictions import ModelPrediction

from typing import Union
from PIL.Image import Image

class ModelInterface():
    """
        Interfaz común para todos los modelos del Model Loader.
    """

    def _init(self) -> None:
        """
            Inicializa el modelo.
        """
        pass

    def get_name(self) -> str:
        """
            Devuelve el nombre del modelo.
        """
        pass

    def predict(self, image_path: Union[Image, str]) -> tuple[ModelPrediction,list[ModelPrediction]]:
        """
            Realiza una predicción sobre una imagen.
        """
        pass

