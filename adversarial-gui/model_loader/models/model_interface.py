# Autor: José Luis López Ruiz
# Fecha: 08/05/2024
# Descripción: Este script define la Interfaz común para todos los modelos del Model Loader.

from model_loader.model_utils.model_predictions import ModelPrediction

class ModelInterface():

    def _init(self) -> None:
        pass

    def get_name(self) -> str:
        pass

    def predict(self, image_path: str) -> tuple[ModelPrediction,list[ModelPrediction]]:
        pass



# class CustomModelInterface(ModelInterface):
#     def __init__(self, model_path: str) -> None:
#         pass
    
#     def predict(self, image_path: str) -> tuple[ModelPrediction,list[ModelPrediction]]:
#         pass