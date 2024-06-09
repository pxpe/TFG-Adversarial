# Autor: José Luis López Ruiz
# Fecha: 23/05/2024
# Descripción: Este script define la clase AdversarialPurification que implementa la defensa Purificación Adversaria en una imagen.

import tensorflow as tf
from tensorflow import Tensor
import os
import numpy as np

from tensorflow.keras.models import load_model

class AdversarialPurification():
    """
        Clase que implementa la defensa Purificación Adversaria en una imagen.

    """
    DENOISER_AUTOENCODER_PATH = os.path.abspath(os.path.curdir + "/adversarial-gui/default_models/denoiser_autoencoder.h5")
    #DENOISER_AUTOENCODER_PATH = os.path.abspath(os.path.curdir + "/default_models/denoiser_autoencoder.h5")

    AUTOENCODER_INPUT_SHAPE = [1,32,32,3]
    AUTOENCODER_SIZE = 32

    def __init__(self, noisy_image: Tensor) -> Tensor:
        self.noisy_image = noisy_image
        self.__denoiser = load_model(self.DENOISER_AUTOENCODER_PATH)
        self.purified_image = self.__purify()
        

    def __purify(self) -> Tensor:
        """
            Aplica la defensa Purificación Adversaria en una imagen y devuelve la imagen purificada.
        """
        resized_noisy_image = tf.image.resize(self.noisy_image, (self.AUTOENCODER_SIZE, self.AUTOENCODER_SIZE))
        perturbated_img_resized = tf.reshape(resized_noisy_image, self.AUTOENCODER_INPUT_SHAPE)
        purified_image = self.__denoiser.predict(perturbated_img_resized)
        return purified_image
    
    def get_purified_image(self) -> Tensor:
        """
            Devuelve la imagen purificada.
        """
        return self.purified_image
