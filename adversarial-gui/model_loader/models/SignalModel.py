# Autor: José Luis López Ruiz
# Fecha: 08/05/2024
# Descripción: Este.


from model_loader.model_utils.model_predictions import ModelPrediction
from model_loader.model_utils.model_exceptions import ModelNotLoadedException
from .model_interface import ModelInterface

import tensorflow as t
from keras.preprocessing import image
import numpy as n
from typing import List
import os

class ModelSignalModel(ModelInterface):

    CLASES = {
        0: 'Señal Máx. 120Km/h',
        1: 'Señal Máx. 50Km/h',
        2: 'Señal Radar',
        3: 'Señal STOP'
    }

    MODEL_PATH = os.path.join(os.curdir,"default_models", "signal_model.h5")

    def __init__(self) -> None:
        print(self.MODEL_PATH)
        print(os.path.curdir)
        try:
            
            self.model = t.keras.models.load_model(self.MODEL_PATH)
            self.model_name = 'SignalModel'
        except Exception as e:
            print(e)
            raise ModelNotLoadedException()

    def get_name(self) -> str:
        return self.model_name
    
    def predict(self, image_path: str) -> List[ModelPrediction]:
        img = image.image_utils.load_img(image_path, target_size=(224, 224))
        img = image.image_utils.img_to_array(img)

        yhat = self.model.predict(n.expand_dims(img, axis=0))
        
        result = []
        print(yhat)
        for i, clase in enumerate(yhat[0]):
            result.append(ModelPrediction(self.CLASES[i], str(clase)+'%'))
        return result
    
