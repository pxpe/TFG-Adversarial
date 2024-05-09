# Autor: José Luis López Ruiz
# Fecha: 09/05/2024
# Descripción:  Este script define las factorias concretas para generar los modelos, la cual implementa la interfaz AbstractModelFactory.


from .model_factory import AbstractModelFactory

from .model_interface import ModelInterface
from .implementations.ResNet50 import ModelResNet50
from .implementations.SignalModel import ModelSignalModel


class ResNet50Factory(AbstractModelFactory):
    """
        Factoría concreta para crear el modelo ResNet50.
    """
    def create_model(self) -> ModelInterface:
        return ModelResNet50()
    
class SignalModelFactory(AbstractModelFactory):
    """
        Factoría concreta para crear el modelo SignalModel.
    """
    def create_model(self) -> ModelInterface:
        return ModelSignalModel()