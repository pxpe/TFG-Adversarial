# Autor: José Luis López Ruiz
# Fecha: 04/05/2024
# Descripción: Este script contiene las clases de las excepciones correspondientes al modelo.


# Clase de excepción para cuando se intenta cargar un modelo inválido
class InvadidModelName(Exception):
    def __init__(self, model_name : str):
        super().__init__(f"Modelo inválido: {model_name}")

class ModelNotLoadedException(Exception):
    def __init__(self):
        super().__init__('Modelo no cargado')