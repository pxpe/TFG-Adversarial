# Autor: José Luis López Ruiz
# Fecha: 08/05/2024
# Descripción: Este script define la Interfaz común para todos los modelos del Model Loader.

from model_loader.model_utils.model_predictions import ModelPrediction

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

    def predict(self, image_path: str) -> tuple[ModelPrediction,list[ModelPrediction]]:
        """
            Realiza una predicción sobre una imagen.
        """
        pass



# class CustomModelInterface(ModelInterface):
#     def __init__(self, model_path: str) -> None:
#         pass
    
#     def predict(self, image_path: str) -> tuple[ModelPrediction,list[ModelPrediction]]:
#         pass