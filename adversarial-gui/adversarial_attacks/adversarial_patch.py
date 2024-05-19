# Autor: José Luis López Ruiz
# Fecha: 19/05/2024
# Descripción: Este script tiene la clase AdversarialPatch que implementa el ataque Parche Adversario en una imagen.

from model_loader.models.model_interface import ModelInterface

import tensorflow as tf
from keras.preprocessing import image
import cv2
import numpy as n
import foolbox as fb
import os
from tensorflow import Tensor

class AdversarialPatch():

    """
    Clase que implementa el ataque Parche Adversario en una imagen.

    Atributos:
    - source_image_path: Ruta de la imagen original.
    - objective_class_label: Etiqueta de la clase objetivo.
    - model: Modelo utilizado para el ataque.
    - patch_size: Tamaño del parche.
    - adversarial_patch: Parche adversario generado.
    - PATCH_CENTER: Centro del parche en la imagen.
    - optimizer: Optimizador utilizado para el ataque.
    - PREPROCESSING: Preprocesamiento necesario para el modelo Foolbox.
    - loss_object: Función de pérdida utilizada para el ataque.
    - pretrained_model: Modelo preentrenado utilizado para el ataque.
    - fmodel: Modelo compatible con Foolbox.
    - adversarial_image: Imagen adversaria generada.

    """

    MEAN = tf.constant([0.485, 0.456, 0.406])
    STD = tf.constant([0.229, 0.224, 0.225])

    LEARNING_RATE = 0.1
    ITERATIONS = 1

    def __init__(self, source_image_path: str, target_class_label : int, model : ModelInterface, patch_size : int = 50) -> None:
        self.model = model
        self.source_image = self.__preprocess_image(source_image_path)

        self.target_class_label = target_class_label

        self.patch_size = patch_size
        self.adversarial_patch = n.random.rand(patch_size, patch_size, 3) # Inicializar el parche con ruido aleatorio
        self.adversarial_patch = tf.Variable(self.adversarial_patch, dtype=tf.float32)
        self.PATCH_CENTER = (n.random.randint(0, 224), n.random.randint(0, 224))


        self.optimizer = tf.optimizers.Adam(self.LEARNING_RATE)

        self.PREPROCESSING = {'mean': self.MEAN, 'std': self.STD}
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.pretrained_model = model.get_model()
        
        # Convertir el modelo de Keras a un modelo compatible con Foolbox
        self.fmodel = fb.TensorFlowModel(self.pretrained_model, bounds=(0, 1), preprocessing=self.PREPROCESSING)
        
        self.__generate_adversarial_patch()
        self.adversarial_image = self.__generate_adversarial_image()
        self.__save_adversarial_image()

    def __preprocess_image(self, image_path: str) -> Tensor:
        image_raw = tf.io.read_file(image_path)
        image = tf.image.decode_image(image_raw)
        image = tf.cast(image, tf.float32)
        image = self.model.resize_image(image)
        image = image[None, ...]

        image = self.model.preprocess_image(image)

        return image
    

    # Función para aplicar el parche a la imagen
    def __apply_patch(self, image, patch, center):
        patch_shape = tf.shape(patch)
        start_x = center[0] - patch_shape[0] // 2
        start_y = center[1] - patch_shape[1] // 2

        # Create a mask to insert the patch
        mask = tf.zeros_like(image)
        mask = tf.tensor_scatter_nd_update(
            mask,
            indices=[[0, start_x + i, start_y + j, k] for i in range(patch_shape[0]) for j in range(patch_shape[1]) for k in range(patch_shape[2])],
            updates=tf.reshape(patch, [-1])
        )

        patched_image = image * (1 - mask) + mask
        return patched_image
        
    def __generate_adversarial_patch(self):
        for i in range(self.ITERATIONS):
            with tf.GradientTape() as tape:
                tape.watch(self.adversarial_patch)
                patched_image = self.__apply_patch(self.source_image, self.adversarial_patch, self.PATCH_CENTER)
                preds = self.fmodel(patched_image)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[self.target_class_label], logits=preds)
            
            grads = tape.gradient(loss, self.adversarial_patch)
            if grads is not None:
                self.optimizer.apply_gradients([(grads, self.adversarial_patch)])
            else:
                print('No se encontraron gradientes.')
            
            # Clip del parche para mantenerlo en el rango válido
            self.adversarial_patch.assign(tf.clip_by_value(self.adversarial_patch, -3, 3))  # Ajusta según el rango de tu imagen preprocesada
            
            print(f'Iteración {i}, Pérdida: {n.mean(loss.numpy())}')
    
    
    def __generate_adversarial_image(self) -> Tensor:
        self.adversarial_image = self.__apply_patch(self.source_image, self.adversarial_patch, self.PATCH_CENTER)

    def __save_adversarial_image(self) -> None:
        image = self.adversarial_image.numpy()
        image = image[0]
        image = image * 255
        image = image.astype(n.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join('generated_patches',f'{self.model.get_name()}:{self.target_class_label}'), image)

    def get_adversarial_image(self) -> Tensor:
        return self.adversarial_image
    
    def get_adversarial_patch(self) -> Tensor:
        patch_display = (self.adversarial_patch.numpy() * self.PREPROCESSING['std'] + self.PREPROCESSING['mean']) * 255
        patch_display = n.clip(patch_display, 0, 255).astype(n.uint8)
        return patch_display
    
    def get_source_image(self) -> Tensor:
        return self.source_image
    