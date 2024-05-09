# Autor: José Luis López Ruiz
# Fecha: 08/05/2024
# Descripción: Este script define la clase ModelSignalModel, la cual implementa la interfaz ModelInterface y representa un modelo de red neuronal creada, signal_model.h5.


from model_loader.model_utils.model_predictions import ModelPrediction
from model_loader.model_utils.model_exceptions import ModelNotLoadedException
from ..model_interface import ModelInterface

from model_loader.model_utils.model_singleton import Singleton


import tensorflow as t
from keras.preprocessing import image
import numpy as n
from typing import List
import os
import cv2

@Singleton
class ModelSignalModel(ModelInterface):
    """
        Clase que implementa el modelo SignalModel.
        Clases del modelo:
        - 0: Señal Máx. 120Km/h
        - 1: Señal Máx. 50Km/h
        - 2: Señal Radar
        - 3: Señal STOP
    """

    CLASES = {
        0: 'Señal Máx. 120Km/h',
        1: 'Señal Máx. 50Km/h',
        2: 'Señal Radar',
        3: 'Señal STOP'
    }

    MODEL_PATH = os.path.abspath(os.path.curdir + "/default_models/signal_model.h5")

    def __init__(self) -> None:
        print(self.MODEL_PATH)
        try:
            self.model = t.keras.models.load_model(self.MODEL_PATH)
            self.model_name = 'SignalModel'
        except Exception as e:
            print(e)
            raise ModelNotLoadedException()

    def get_name(self) -> str:
        return self.model_name
    
    def predict(self, image_path: str) -> List[ModelPrediction]:
        img = cv2.imread(image_path)
        resize = t.image.resize(img, [256, 256])

        yhat = self.model.predict(n.expand_dims(resize/255, axis=0))
        
        predictions = []
        for i, clase in enumerate(yhat[0]):
            predictions.append(ModelPrediction(self.CLASES[i], str(round(clase * 100,2))))

        result_index = n.argmax(yhat)
        result = ModelPrediction(self.CLASES[result_index], str(round(yhat[0][result_index]*100,2)))

        return [result, predictions]
    
