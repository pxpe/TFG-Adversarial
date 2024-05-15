# Autor: José Luis López Ruiz
# Fecha: 08/05/2024
# Descripción: Este script define la clase ModelResNet50, la cual implementa la interfaz ModelInterface y representa un modelo de red neuronal ResNet50.


from model_loader.model_utils.model_predictions import ModelPrediction
from model_loader.model_utils.model_labels import getImageNetLabelsToIndex

from ..model_interface import ModelInterface

from keras.preprocessing import image
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input
from keras.applications.resnet import decode_predictions

import tensorflow as tf
from tensorflow import Tensor

import numpy as n

from typing import Union
from PIL.Image import Image

from model_loader.model_utils.model_singleton import Singleton

@Singleton
class ModelResNet50(ModelInterface):
    """
        Clase que implementa el modelo ResNet50.
    """

    def __init__(self) -> None:
        self.model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        self.model_name = "ResNet50"

    def get_name(self) -> str:
        return self.model_name
    
    def predict(self, image_path: Union[Image, str], not_decoded : bool = False) -> Union[tuple[ModelPrediction,list[ModelPrediction]] , n.ndarray]:
        if type(image_path) == str:
            img = image.image_utils.load_img(image_path, target_size=(224, 224))
        else:
            img = image.image_utils.img_to_array(image_path)

        x = preprocess_input(n.expand_dims(img, axis=0))
        preds = self.model.predict(x)
        if not_decoded:
            return preds
        
        decoded_preds = decode_predictions(preds, top=5)[0]
        return [ModelPrediction(decoded_preds[0][1],str(round(decoded_preds[0][2] * 100,2))) , [ModelPrediction(p[1],str(round(p[2]*100,2))) for p in decoded_preds]]
    
    def get_label(self, class_str: str) -> Tensor:

        class_str = class_str.replace("_", " ")
        class_index = getImageNetLabelsToIndex()[class_str]
        label = tf.one_hot(class_index, 1000)
        label = tf.reshape(label, (1, 1000))

        return label
    
    def get_model(self) -> ResNet50:
        return self.model