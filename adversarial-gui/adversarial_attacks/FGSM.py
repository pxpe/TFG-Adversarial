
from model_loader.models.model_interface import ModelInterface

import tensorflow as tf
from keras.preprocessing import image

from tensorflow import Tensor

from PIL.Image import Image

class FGSMAttack():

    def __init__(self, source_image_path: str, epsilon : float, input_label: str, model : ModelInterface) -> None:
        self.source_image = self.__preprocess_image(source_image_path)

        self.epsilon = epsilon
        self.model = model

        self.input_label = input_label
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.pretrained_model = model.get_model()
        
        self.adversarial_pattern = self.__generate_adversarial_pattern()
        self.adversarial_image = self.__generate_adversarial_image()

    def __preprocess_image(self, image_path: str) -> Tensor:
        image_raw = tf.io.read_file(image_path)
        image = tf.image.decode_image(image_raw)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (224, 224))
        image = image[None, ...]
        return image

    def __generate_adversarial_image(self):
        adv_img = self.source_image + (self.epsilon * self.adversarial_pattern)
        adv_img = tf.clip_by_value(adv_img, -1, 1)
        return adv_img

    def __generate_adversarial_pattern(self):
        with tf.GradientTape() as tape:
            tape.watch(self.source_image)
            prediction = self.pretrained_model(self.source_image)
            loss = self.loss_object(self.input_label, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, self.source_image)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        
        return signed_grad
    
    def get_adversarial_image(self) -> Tensor:
        return self.adversarial_image
    
    def get_adversarial_pattern(self) -> Tensor:
        return self.adversarial_pattern
    
    def get_source_image(self) -> Tensor:
        return self.source_image
    
