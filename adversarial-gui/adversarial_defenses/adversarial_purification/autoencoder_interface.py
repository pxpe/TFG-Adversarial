# Autor: José Luis López Ruiz
# Fecha: 29/05/2024
# Descripción: Este script define la clase AdversarialPurification que implementa la defensa Purificación Adversaria en una imagen.


import os
from tensorflow import Tensor

class DenoiserAutoencoder():
    """
        Interfaz para aplicar la defensa Purificación Adversaria, reduciendo el ruido en una imagen.
    """

    DENOISER_AUTOENCODER_PATH = None   # Ruta del modelo autoencoder
    AUTOENCODER_INPUT_SHAPE = None     # Formato de entrada del autoencoder
    AUTOENCODER_SIZE = None            # Tamaño de la imagen de entrada del autoencoder


    def __init__(self, noisy_image: Tensor) -> None:
        """
            Constructor que inicializa la defensa Purificación Adversaria.
            Parámetros:
                - noisy_image: Imagen ruidosa.
        """
        pass


    def __purify(self) -> Tensor:
        """
            Aplica la defensa Purificación Adversaria en una imagen y devuelve la imagen purificada.

            Devuelve:
                - Imagen purificada.
        """
        pass

    
    def get_purified_image(self) -> Tensor:
        """
            Devuelve la imagen purificada.
        """
        return self.purified_image

    def get_autoencoder_input_shape(self) -> list:
        """
            Devuelve el formato de entrada del autoencoder.
        """
        return self.AUTOENCODER_INPUT_SHAPE
    
    def get_autoencoder_size(self) -> int:
        """
            Devuelve el tamaño de la imagen de entrada del autoencoder.
        """
        return self.AUTOENCODER_SIZE