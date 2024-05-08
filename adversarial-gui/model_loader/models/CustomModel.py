# Autor: José Luis López Ruiz
# Fecha: 08/05/2024
# Descripción: Este.


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
    
    def predict(self, image_path: str) -> List[ModelPrediction]:
        img = image.image_utils.load_img(image_path, target_size=(224, 224))
        img = image.image_utils.img_to_array(img)

        yhat = self.modelo.predict(n.expand_dims(img, axis=0))
        return [ModelPrediction("Custom",str(yhat[0]))]

    
