# Autor: José Luis López Ruiz
# Fecha: 04/05/2024
# Descripción: Este script contiene la clase ModelLoader, la cual se encarga de cargar un modelo de red neuronal.

from model_loader.model_utils.model_exceptions import InvadidModelName, ModelNotLoadedException
from model_loader.model_utils.model_predictions import ModelPrediction

from model_loader.models.concrete_model_factories import ResNet50Factory, SignalModelFactory

from model_loader.models.implementations.ResNet50 import ModelResNet50
from model_loader.models.implementations.SignalModel import ModelSignalModel

class ModelLoader(): 

    def __init__(self, default_model: str = "ResNet50"): 
        """
            Inicializa el cargador de modelos, instanciando ResNet50.
            Parametros:	
            -    default_model (str): Nombre del modelo por defecto a cargar.
        """
        self.models = {}
        self.model = None
        
        # Cargar modelo por defecto
        self.init_model(default_model)

    def init_model(self, str_model: str):
        """
            Inicializa un modelo de red neuronal.
            Parametros:	
            -    str_model (str): Nombre del modelo a cargar.
        """
        print(f"Cargando modelo: {str_model}")
        if not str_model:
            raise InvadidModelName(model_name=str_model)
        
        if str_model == "ResNet50": 
            #self.model = ResNet50Factory()
            self.model = ModelResNet50()
            print(f"Modelo cargado: {self.model.get_name()}")

        elif str_model == "SignalModel":
            #self.model = SignalModelFactory()
            self.model = ModelSignalModel()
            print(f"Modelo cargado: {self.model.get_name()}")
        else:
            raise InvadidModelName(model_name=str_model)
        
        self.models.update({self.model.get_name() : self.model})

    
    def switch_model(self, str_model: str):
        """
            Cambia el modelo de red neuronal a utilizar.
            Parametros:	
            -    str_model (str): Nombre del modelo a cargar.
        """
        print(f"Cargando modelo: {str_model}")
        try:
            self.model = self.models[str_model]
        except KeyError:
            self.init_model(str_model)
            
        print(f"Modelo cargado: {self.model.get_name()}")


    def predict(self, image: str) -> tuple[ModelPrediction,list[ModelPrediction]]:
        """
            Realiza una predicción sobre una imagen usando el modelo cargado.
            Parametros:	
            -    image (str): Ruta de la imagen a predecir.
        """
        if self.model is None:
            raise ModelNotLoadedException()
        else:
            return self.model.predict(image)
        