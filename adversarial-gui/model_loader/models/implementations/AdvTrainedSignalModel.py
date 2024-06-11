# Autor: José Luis López Ruiz
# Fecha: 08/05/2024
# Descripción: Este script define la clase ModelSignalModel, la cual implementa la interfaz ModelInterface y representa un modelo de red neuronal creada, signal_model.h5.


from model_loader.model_utils.model_predictions import ModelPrediction
from model_loader.model_utils.model_exceptions import ModelNotLoadedException
from model_loader.model_utils.model_labels import getSignalModelLabelsToIndex, getSignalModelIndexToLabels

from ..model_interface import ModelInterface

from keras.preprocessing import image

import tensorflow as t
import numpy as n
import os
import cv2
from typing import Union
from PIL.Image import Image
from PIL import Image as im


import tensorflow as tf
from tensorflow import Tensor

from keras.applications import imagenet_utils

from model_loader.model_utils.model_singleton import Singleton

@Singleton # Aplicamos el patrón de diseño Singleton mediante un decorador, este patrón nos permite tener una única instancia de la clase ModelInterface y de sus subclases.
class ModelAdvTrainedSignalModel(ModelInterface):
    """
        Clase que implementa el modelo SignalModel.
        Clases del modelo:
        - 0: Señal Máx. 120Km/h
        - 1: Señal Máx. 50Km/h
        - 2: Señal Radar
        - 3: Señal STOP
    """

    MODEL_PATH = os.path.abspath(os.path.curdir + "/adversarial-gui/default_models/SignAdversaryTrainedClassifier.h5")
    #MODEL_PATH = os.path.abspath(os.path.curdir + "/default_models/SignAdversaryTrainedClassifier.h5")
    
    def __init__(self) -> None:
        self.CLASES = getSignalModelIndexToLabels()
        try:
            self.model = t.keras.models.load_model(self.MODEL_PATH)
            self.model_name = 'AdvTrainedSignalModel'
        except Exception as e:
            print(e)
            raise ModelNotLoadedException()

    def get_name(self) -> str:
        return self.model_name
    
    def predict(self, image_path: Union[Image,Tensor, str], not_decoded : bool = False) -> Union[tuple[ModelPrediction,list[ModelPrediction]] , n.ndarray]:
        if type(image_path) == str:
            img = cv2.imread(image_path)
            img = t.image.resize(img, [256, 256])
        elif type(image_path) == Image:
            img = image.image_utils.img_to_array(image_path)
        else:
            img = image_path[0].numpy()

        preds = self.model.predict(n.expand_dims(img/255, axis=0))
        
        if not_decoded:
            return preds

        predictions = []
        for i, clase in enumerate(preds[0]):
            predictions.append(ModelPrediction(self.CLASES[i], str(round(clase * 100,2))))

        result_index = n.argmax(preds)
        result = ModelPrediction(self.CLASES[result_index], str(round(preds[0][result_index]*100,2)))

        return [result, predictions]
    
    def preprocess_image(self, image: Tensor) -> Tensor:
        return imagenet_utils.preprocess_input(
            image, data_format=None, mode="tf"
        )

    def normalize_image(self, image: Tensor) -> Image:
        image = image.numpy()
        img_original = im.fromarray(((image[0] + 1) * 127.5).astype("uint8"))
        return img_original
    
    def normalize_patch(self, patch: Tensor) -> Image:
        patch = patch.numpy()
        img_original = im.fromarray(((patch + 1) * 127.5).astype("uint8"))
        return img_original

    def resize_image(self, image: Tensor) -> Tensor:
        return t.image.resize(image, [256, 256])
    
    def reshape_image(self, image: Tensor) -> Tensor:
        return tf.reshape(image, (1, 256, 256, 3))

    def get_label(self, class_str: str) -> Tensor:

        class_index = getSignalModelLabelsToIndex()[class_str]
        label = tf.one_hot(class_index, len(self.CLASES))
        label = tf.reshape(label, (1, len(self.CLASES)))

        return label
    
    def get_classes(self) -> list[str]:
        return getSignalModelLabelsToIndex()

    def get_model(self) -> t.keras.Model:
        return self.model