# Autor: José Luis López Ruiz
# Fecha: 30/05/2024
# Descripción: Este script define el Model Factory, el cual se encarga de crear los modelos de red neuronal de forma genérica.

from model_loader.models.model_interface import ModelInterface
from .model_exceptions import InvadidModelName

from model_loader.models.implementations.ResNet50V2 import ModelResNet50V2
from model_loader.models.implementations.SignalModel import ModelSignalModel
from model_loader.models.implementations.AdvTrainedSignalModel import ModelAdvTrainedSignalModel
from model_loader.models.implementations.MobileNetV2 import ModelMobileNetV2
from model_loader.models.implementations.VGG19 import ModelVGG19



class ModelFactory():
    """
        Factoría de modelos de red neuronal.
        Métodos:
        -   create_model(model_name: str) -> ModelInterface: Crea un modelo de red neuronal a partir de su nombre.

    """

    def __init__(self) -> None:
        pass

    def create_model(self, model_name : str) -> ModelInterface:
        """
            Inicializa un modelo de red neuronal.
            Parametros:	
            -    str_model (str): Nombre del modelo a cargar.
        """
        model = None

        if model_name == "ResNet50 V2": 
            model = ModelResNet50V2()
        elif model_name == "SignalModel":
            model = ModelSignalModel()
        elif model_name == "MobileNet V2":
            model = ModelMobileNetV2()
        elif model_name == "VGG19":
            model = ModelVGG19()
        elif model_name == "AdvTrainedSignalModel":
            model = ModelAdvTrainedSignalModel()
        else:
            raise InvadidModelName(model_name=model_name)

        return model