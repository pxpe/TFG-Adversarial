# Autor: José Luis López Ruiz
# Fecha: 04/05/2024
# Descripción: Este script contiene las clases de las excepciones correspondientes al modelo.


class InvadidModelName(Exception):
    """
        Clase de excepción para cuando se intenta cargar un modelo inválido.
    """
    def __init__(self, model_name : str):
        super().__init__(f"Modelo inválido: {model_name}")

class ModelNotLoadedException(Exception):
    """
        Clase de excepción para indicar que un modelo no se ha cargado.
    """
    def __init__(self):
        super().__init__('Modelo no cargado')