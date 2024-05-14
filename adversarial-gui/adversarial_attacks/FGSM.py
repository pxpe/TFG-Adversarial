
from attack_interface import AdversarialAttackInterface
from ..model_loader.models.model_interface import ModelInterface

import tensorflow as tf

from PIL.Image import Image

class FGSMAttack():

    def __init__(self, source_image_path: str, epsilon : float,model : ModelInterface) -> None:
        self.source_image_path = source_image_path
        self.source_image = tf.image.load_img(source_image_path, target_size=(224, 224))

        self.epsilon = epsilon
        self.model = model

        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        
        self.adversarial_pattern = self.__generate_adversarial_pattern()
        self.adversarial_image = self.__generate_adversarial_image()


    def __generate_adversarial_image(self) -> None:
        self.adversarial_image = self.source_image + self.epsilon * self.adversarial_pattern
        self.adversarial_image = tf.clip_by_value(self.adversarial_image, -1, 1)

    def __generate_adversarial_pattern(self, input_label: int) -> None:
        with tf.GradientTape() as tape:
            tape.watch(self.source_image)
            prediction = self.model.predict(self.source_image)
            loss = self.loss_object(input_label, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, self.source_image)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        return signed_grad
    
    def get_adversarial_image(self) -> Image:
        return self.adversarial_image
    
    def get_adversarial_pattern(self) -> Image:
        return self.adversarial_pattern

    def get_source_image(self) -> Image:
        return self.source_image
