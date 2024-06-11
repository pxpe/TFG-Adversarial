# Autor: José Luis López Ruiz
# Fecha: 04/05/2024
# Descripción: Este script contiene la clase ModelLoader, la cual se encarga de cargar un modelo de red neuronal.

from model_loader.model_utils.model_predictions import ModelPrediction
from model_loader.model_utils.model_factory import ModelFactory

from model_loader.model_utils.model_exceptions import ModelNotLoadedException, InvadidModelName


from tensorflow import Tensor

from typing import Union
from PIL.Image import Image

import numpy as n

class ModelLoader(): 

    def __init__(self, default_model: str = "ResNet50 V2"): 
        """
            Inicializa el cargador de modelos, instanciando por defecto ResNet50 V2.
            Parametros:	
            -    default_model (str): Nombre del modelo por defecto a cargar.
        """
        self.models = {}
        self.model = None
        self.factory = ModelFactory()
        
        # Cargar modelo por defecto
        self.switch_model(default_model)

    
    def switch_model(self, str_model: str):
        """
            Cambia el modelo de red neuronal a utilizar.
            Parametros:	
            -    str_model (str): Nombre del modelo a cargar.
        """
        print(f"Cargando modelo: {str_model}")
        try:
            self.model = self.factory.create_model(str_model)        
            print(f"Modelo cargado: {self.model.get_name()}")
        except InvadidModelName as e:
            print("Error al cargar el modelo: ", e)
            


    def predict(self, image_path: Union[Image,Tensor, str], not_decoded : bool = False) -> Union[tuple[ModelPrediction,list[ModelPrediction]] , n.ndarray]:
        """
            Realiza una predicción sobre una imagen usando el modelo cargado.
            Parametros:	
            -    image_path (str | Image: Ruta de la imagen a predecir.
        """
        if self.model is None:
            raise ModelNotLoadedException()
        else:
            return self.model.predict(image_path=image_path, not_decoded=not_decoded)
    
    def preprocess_image(self, image: Image) -> Tensor:
        """
            Preprocesa una imagen para ser usada por el modelo.
            Parametros:	
            -    image (Image): Imagen a preprocesar.
        """
        if self.model is None:
            raise ModelNotLoadedException()
        else:
            return self.model.preprocess_image(image)
        
    def normalize_image(self, image: Tensor) -> Image:
        """
            Normaliza una imagen.
            Parametros:	
            -    image (Tensor): Imagen a normalizar.
        """
        if self.model is None:
            raise ModelNotLoadedException()
        else:
            return self.model.normalize_image(image)
    
    def normalize_patch(self, patch: Tensor) -> Image:
        """
            Normaliza una imagen.
            Parametros:	
            -    patch (Tensor): Parche a normalizar.
        """
        if self.model is None:
            raise ModelNotLoadedException()
        else:
            return self.model.normalize_patch(patch)

    def resize_image(self, image: Tensor) -> Tensor:
        """
            Redimensiona una imagen al tamaño óptimo del modelo.
            Parametros:	
            -    image (Tensor): Imagen a redimensionar.
        """
        if self.model is None:
            raise ModelNotLoadedException()
        else:
            return self.model.resize_image(image)
        
    def reshape_image(self, image: Tensor) -> Tensor:
        """
            Redimensiona una imagen al tamaño óptimo del modelo.
            Parametros:	
            -    image (Tensor): Imagen a redimensionar.
        """
        if self.model is None:
            raise ModelNotLoadedException()
        else:
            return self.model.reshape_image(image)


    def get_label(self, class_str: str) -> Tensor:
        """
            Obtiene la etiqueta de una clase.
            Parametros:
            -    clase (str): Nombre de la clase.
        """
        if self.model is None:
            raise ModelNotLoadedException()
        else:
            return self.model.get_label(class_str)
        
    def get_classes(self) -> dict:
        """
            Obtiene las clases del modelo: { str(class_name) : int(class_id) }.
        """
        if self.model is None:
            raise ModelNotLoadedException()
        else:
            return self.model.get_classes()
    
    def get_model(self) -> object:
        """
            Obtiene el modelo de red neuronal cargado.
        """
        if self.model is None:
            raise ModelNotLoadedException()
        else:
            return self.model.get_model()
    
    def get_name(self) -> str:
        """
            Obtiene el nombre del modelo de red neuronal cargado.
        """
        if self.model is None:
            raise ModelNotLoadedException()
        else:
            return self.model.get_name()