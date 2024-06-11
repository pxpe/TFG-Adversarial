# Autor: José Luis López Ruiz
# Fecha: 29/05/2024
# Descripción: Este script define la clase AdversarialPurification que implementa la defensa Purificación Adversaria en una imagen.


import os
from tensorflow import Tensor
from typing import List

class DenoiserAutoencoder():
    """
    Interfaz común para todos los denoising autoencoders.
    """

    DENOISER_AUTOENCODER_PATH: str = None  # Ruta del modelo autoencoder
    AUTOENCODER_INPUT_SHAPE: List[int] = None  # Formato de entrada del autoencoder
    AUTOENCODER_SIZE: int = None  # Tamaño de la imagen de entrada del autoencoder

    def __init__(self, noisy_image: Tensor) -> None:
        """
        Constructor que inicializa la defensa Purificación Adversaria.
        Parámetros:
            - noisy_image: Imagen ruidosa.
        """
        self.noisy_image = noisy_image
        self.purified_image = None

    def _purify(self) -> Tensor:
        """
        Aplica la defensa Purificación Adversaria en una imagen y devuelve la imagen purificada.

        Devuelve:
            - Imagen purificada.
        """
        raise NotImplementedError("El método _purify debe ser implementado por la subclase.")

    def get_purified_image(self) -> Tensor:
        """
        Devuelve la imagen purificada.
        """
        if self.purified_image is None:
            self.purified_image = self._purify()
        return self.purified_image

    def get_autoencoder_input_shape(self) -> List[int]:
        """
        Devuelve el formato de entrada del autoencoder.
        """
        return self.AUTOENCODER_INPUT_SHAPE

    def get_autoencoder_size(self) -> int:
        """
        Devuelve el tamaño de la imagen de entrada del autoencoder.
        """
        return self.AUTOENCODER_SIZE