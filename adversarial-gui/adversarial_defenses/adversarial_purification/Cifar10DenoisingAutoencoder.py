# Autor: José Luis López Ruiz
# Fecha: 23/05/2024
# Descripción: Este script define la clase Cifar10DenoiserAutoEncoder que implementa la defensa Purificación Adversaria en una imagen e implementa la interfaz DenoiserAutoencoder.



import os
import numpy as np
from tensorflow import Tensor
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

from image_utils.utilities import divide_image, reconstruct_image

from .autoencoder import DenoiserAutoencoder

class Cifar10DenoisingAutoEncoder(DenoiserAutoencoder):
    """
    Clase que implementa la defensa Purificación Adversaria en una imagen utilizando un Autoencoder entrenado con el dataset CIFAR-10 de imágenes 32x32.
    Implementa la interfaz DenoiserAutoencoder.
    """

    DENOISING_AUTOENCODER_PATH = os.path.abspath(os.path.curdir + "/default_models/denoiser_autoencoder.h5")
    AUTOENCODER_INPUT_SHAPE = [1, 32, 32, 3]
    AUTOENCODER_SIZE = 32

    def __init__(self, noisy_image: tf.Tensor) -> None:
        super().__init__(noisy_image)
        self.__denoiser = load_model(self.DENOISING_AUTOENCODER_PATH)
        self.purified_image = self._purify()

    def _purify(self) -> tf.Tensor:
        """
        Aplica la defensa Purificación Adversaria en una imagen y devuelve la imagen purificada.

        Devuelve:
            - Imagen purificada.
        """
        images = divide_image(self.noisy_image, self.noisy_image.shape[0], self.AUTOENCODER_SIZE)

        reconstructed_images = []
        for image_chunk in images:
            reshaped_image_chunk = tf.reshape(image_chunk, self.AUTOENCODER_INPUT_SHAPE)
            reconstructed_img = self.__denoiser.predict(reshaped_image_chunk)
            reconstructed_images.append(reconstructed_img)
            
        reconstructed_image = reconstruct_image(reconstructed_images, self.noisy_image.shape[0], self.AUTOENCODER_SIZE)
        purified_image = cv2.GaussianBlur(reconstructed_image, (3, 3), 5)

        return purified_image