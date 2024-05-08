# Autor: José Luis López Ruiz
# Fecha: 08/05/2024
# Descripción: Este script define la clase ModelResNet50, la cual implementa la interfaz ModelInterface y representa un modelo de red neuronal ResNet50.


from model_loader.model_utils.model_predictions import ModelPrediction
from typing import List

from .model_interface import ModelInterface

from keras.preprocessing import image
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input
from keras.applications.resnet import decode_predictions
import numpy as n


class ModelResNet50(ModelInterface):
    def __init__(self) -> None:
        self.model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        self.model_name = "ResNet50"

    def get_name(self) -> str:
        return self.model_name
    
    def predict(self, image_path: str) -> tuple[ModelPrediction,list[ModelPrediction]]:
        img = image.image_utils.load_img(image_path, target_size=(224, 224))
        img = image.image_utils.img_to_array(img)

        x = preprocess_input(n.expand_dims(img, axis=0))
        preds = self.model.predict(x)
        decoded_preds = decode_predictions(preds, top=5)[0]
        return [ModelPrediction(decoded_preds[0][1],str(round(decoded_preds[0][2] * 100,2))) , [ModelPrediction(p[1],str(round(p[2]*100,2))) for p in decoded_preds]]
    
