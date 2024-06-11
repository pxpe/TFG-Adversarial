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

from .autoencoder_interface import DenoiserAutoencoder

class Cifar10DenoiserAutoEncoder(DenoiserAutoencoder):

    """
        Clase que implementa la defensa Purificación Adversaria en una imagen utilizando un Autoencoder entrenado con el dataset CIFAR-10 de imágenes 32x32.
        Implementa la interfaz DenoiserAutoencoder.
    """

    #DENOISER_AUTOENCODER_PATH = os.path.abspath(os.path.curdir + "/default_models/denoiser_autoencoder.h5") # Ruta del modelo autoencoder
    DENOISER_AUTOENCODER_PATH = os.path.abspath(os.path.curdir + "/adversarial-gui/default_models/denoiser_autoencoder.h5") # Ruta del modelo autoencoder

    AUTOENCODER_INPUT_SHAPE = [1,32,32,3] # Formato de entrada del autoencoder
    AUTOENCODER_SIZE = 32                 # Tamaño de la imagen de entrada del autoencoder

    def __init__(self, noisy_image: Tensor) -> None:
        self.noisy_image = noisy_image
        self.__denoiser = load_model(self.DENOISER_AUTOENCODER_PATH)
        self.purified_image = self.__purify()

    def __purify(self) -> Tensor:

        # Divide la imagen en bloques de 32x32
        images = divide_image(self.noisy_image, self.noisy_image.shape[0], self.AUTOENCODER_SIZE)

        # Procesa cada bloque de la imagen con el autoencoder
        reconstructed_images = []
        for image_chunk in images:
            # Reescala la imagen a 32x32x3
            reshaped_image_chunk = tf.reshape(image_chunk, self.AUTOENCODER_INPUT_SHAPE)
            # Procesa la imagen con el autoencoder
            reconstructed_img = self.__denoiser.predict(reshaped_image_chunk)
            # Añade la imagen procesada a la lista de imágenes reconstruidas
            reconstructed_images.append(reconstructed_img)
            
        # Reconstruye la imagen original a partir de los bloques procesados
        reconstructed_image = reconstruct_image(reconstructed_images, self.noisy_image.shape[0], self.AUTOENCODER_SIZE)

        purified_image = cv2.GaussianBlur(reconstructed_image, (3, 3), 0)

        return purified_image