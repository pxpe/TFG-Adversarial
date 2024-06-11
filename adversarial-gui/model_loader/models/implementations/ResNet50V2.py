# Autor: José Luis López Ruiz
# Fecha: 08/05/2024
# Descripción: Este script define la clase ModelResNet50, la cual implementa la interfaz ModelInterface y representa un modelo de red neuronal ResNet50V2.


from model_loader.model_utils.model_predictions import ModelPrediction
from model_loader.model_utils.model_labels import getImageNetLabelsToIndex

from ..model_interface import ModelInterface

from keras.preprocessing import image
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.resnet_v2 import preprocess_input
from keras.applications.resnet_v2 import decode_predictions

import tensorflow as tf
from tensorflow import Tensor

import numpy as n

from typing import Union
from PIL.Image import Image
from PIL import Image as im

from model_loader.model_utils.model_singleton import Singleton

@Singleton # Aplicamos el patrón de diseño Singleton mediante un decorador, este patrón nos permite tener una única instancia de la clase ModelInterface y de sus subclases.
class ModelResNet50V2(ModelInterface):
    """
        Clase que implementa el modelo ResNet50 V2.
    """

    def __init__(self) -> None:
        self.model = ResNet50V2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        self.model_name = "ResNet50 V2"

    def get_name(self) -> str:
        return self.model_name
    
    def predict(self, image_path: Union[Image,Tensor, str], not_decoded : bool = False) -> Union[tuple[ModelPrediction,list[ModelPrediction]] , n.ndarray]:
        if type(image_path) == str:
            img = image.image_utils.load_img(image_path, target_size=(224, 224))
            img = image.image_utils.img_to_array(img)
            img = preprocess_input(n.expand_dims(img, axis=0))
        else:
            img = image_path

        if not_decoded:
            return self.model(img)

        preds = self.model.predict(img)
        
        decoded_preds = decode_predictions(preds, top=5)[0]
        return [ModelPrediction(decoded_preds[0][1],str(round(decoded_preds[0][2] * 100,2))) , [ModelPrediction(p[1],str(round(p[2]*100,2))) for p in decoded_preds]]
    
    def preprocess_image(self, image: Tensor) -> Tensor:
        return preprocess_input(image)

    def normalize_image(self, image: Tensor) -> Image:
        image = image.numpy()
        img_original = im.fromarray(((image[0] + 1) * 127.5).astype("uint8"))
        return img_original

    def normalize_patch(self, patch: Tensor) -> Image:
        patch = patch.numpy()
        img_original = im.fromarray(((patch + 1) * 127.5).astype("uint8"))
        return img_original

    def resize_image(self, image: Tensor) -> Tensor:
        return tf.image.resize(image, [224, 224])

    def reshape_image(self, image: Tensor) -> Tensor:
        return tf.reshape(image, (1, 224, 224, 3))

    def get_label(self, class_str: str) -> Tensor:
        model_labels = getImageNetLabelsToIndex()
        class_str = class_str.replace('_', ' ')
        class_index = model_labels[class_str]
        label = tf.one_hot(class_index, 1000)
        label = tf.reshape(label, (1, 1000))

        return label
    
    def get_classes(self) -> list[str]:
        return getImageNetLabelsToIndex()
    
    def get_model(self) -> ResNet50V2:
        return self.model