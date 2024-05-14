# Autor: José Luis López Ruiz
# Fecha: 08/05/2024
# Descripción: Este script define la clase MobileNetV2, la cual implementa la interfaz ModelInterface y representa un modelo de red neuronal MobileNetV2.


from model_loader.model_utils.model_predictions import ModelPrediction

from ..model_interface import ModelInterface

from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import decode_predictions
import numpy as n

from typing import Union
from PIL.Image import Image


from model_loader.model_utils.model_singleton import Singleton

@Singleton
class ModelMobileNetV2(ModelInterface):
    """
        Clase que implementa el modelo MobileNetV2.
    """

    def __init__(self) -> None:
        self.model = MobileNetV2(include_top=True, weights='imagenet')
        self.model_name = "MobileNetV2"

    def get_name(self) -> str:
        return self.model_name
    
    def predict(self, image_path: Union[Image, str]) -> tuple[ModelPrediction,list[ModelPrediction]]:
        img = image.image_utils.load_img(image_path, target_size=(224, 224))
        img = image.image_utils.img_to_array(img)

        x = preprocess_input(n.expand_dims(img, axis=0))
        preds = self.model.predict(x)
        decoded_preds = decode_predictions(preds, top=5)[0]
        return [ModelPrediction(decoded_preds[0][1],str(round(decoded_preds[0][2] * 100,2))) , [ModelPrediction(p[1],str(round(p[2]*100,2))) for p in decoded_preds]]
    
