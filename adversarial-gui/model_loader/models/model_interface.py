# Autor: José Luis López Ruiz
# Fecha: 08/05/2024
# Descripción: Este script define la Interfaz común para todos los modelos del Model Loader.

from model_loader.model_utils.model_predictions import ModelPrediction

from typing import Union
from PIL.Image import Image
from tensorflow import Tensor
import numpy as n

from model_loader.model_utils.model_singleton import Singleton

@Singleton # Aplicamos el patrón Singleton mediante un decorador, este patrón nos permite tener una única instancia de la clase ModelInterface y de sus subclases.
class ModelInterface():
    """
        Interfaz común para todos los modelos del Model Loader.
    """

    def _init(self) -> None:
        """
            Inicializa el modelo.
        """
        pass

    def get_name(self) -> str:
        """
            Devuelve el nombre del modelo.
        """
        pass

    def predict(self, image_path: Union[Image,Tensor, str], not_decoded : bool = False) -> Union[tuple[ModelPrediction,list[ModelPrediction]] , n.ndarray]:
        """
            Realiza una predicción sobre una imagen.
            Parametros:
            -    image_path (Union[Image, str]): Ruta de la imagen o imagen.
            -    default_mode (bool): Modo por defecto.

            Retorna:
            Si default_mode es True:
            -    ModelPrediction: Predicción del modelo estandar.
            Si default_mode es False:
            -    tuple[ModelPrediction,list[ModelPrediction]]: Predicción del modelo y lista de predicciones.

        """
        pass

    def preprocess_image(self, image: Tensor) -> Tensor:
        """
            Preprocesa una imagen.
            Parametros:
            -    image (Tensor): Imagen a preprocesar.

            Retorna:
            -    Tensor: Imagen preprocesada.
        """
        pass
    
    def normalize_image(self, tensor: Tensor) -> Image:

        """
            Normaliza una imagen.
            Parametros:
            -    tensor (Tensor): Imagen a normalizar.

            Retorna:
            -    Image: Imagen normalizada.
        """
        pass

    def normalize_patch(self, patch: Tensor) -> Image:
        """
            Normaliza un parche.
            Parametros:
            -    patch (Tensor): Parche a normalizar.

            Retorna:
            -    Image: Parche normalizado.
        """
        pass

    def resize_image(self, image: Tensor) -> Tensor:
        """
            Redimensiona una imagen al valor óptimo del modelo.
            Parametros:
            -    image (Tensor): Imagen a redimensionar.
        """
        pass

    def get_label(self, class_str: str) -> Tensor:
        """
            Obtiene la etiqueta de una clase.
            Parametros:
            -    class_str (str): Nombre de la clase.

            Retorna:
            -    Tensor: Etiqueta de la clase.
        """
        pass


    def get_model(self) -> object:
        """
            Devuelve el modelo.
        """
        pass