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

from matplotlib import pyplot as plt

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
    ITERATIONS = 5 if not tf.test.is_gpu_available() else 50

    def __init__(self, source_image_path: str, target_class_label : int, model : ModelInterface, patch_size : int = 50) -> None:
        
        self.model = model
        self.target_class_label = target_class_label

        self.patch_size = patch_size
        self.patch_center = (112, 112)

        self.optimizer = tf.optimizers.Adam(self.LEARNING_RATE)

        self.PREPROCESSING = {'mean': self.MEAN, 'std': self.STD}   
        self.source_image = self.__preprocess_image(source_image_path)

        self.pretrained_model = model.get_model()
        
        # Convertir el modelo de Keras a un modelo compatible con Foolbox
        self.fmodel = fb.TensorFlowModel(self.pretrained_model, bounds=(0, 1), preprocessing=None)
        
        if os.path.exists(os.path.join('adversarial_attacks','generated_patches', f'{self.model.get_name()}_{self.target_class_label}.png')):
            self.adversarial_patch = self.__preprocess_image(os.path.join('adversarial_attacks','generated_patches', f'{self.model.get_name()}_{self.target_class_label}.png'))
        else:
            self.adversarial_patch = self.__generate_adversarial_patch()

        self.adversarial_image = self.__generate_adversarial_image()

        #self.__save_adversarial_patch()

    def __preprocess_image(self, image_path: str) -> Tensor:
        image_raw = tf.io.read_file(image_path)
        image = tf.image.decode_image(image_raw)
        image = tf.cast(image, tf.float32)
        image = self.model.resize_image(image)
        image = image[None, ...]

        image = self.model.preprocess_image(image)

        return image
    

    # Función para aplicar el parche a la imagen
    def __apply_patch(self,image, patch, center):
        image_shape = tf.shape(image)
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

    def __generate_adversarial_patch(self) -> Tensor:
        patch = n.random.rand(self.patch_size, self.patch_size, 3) # Inicializar el parche con ruido aleatorio
        patch = tf.Variable(patch, dtype=tf.float32)

        for i in range(self.ITERATIONS):
            with tf.GradientTape() as tape:
                tape.watch(patch)
                patched_image = self.__apply_patch(self.source_image, patch, self.patch_center)
                preds = self.fmodel(patched_image)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[self.target_class_label], logits=preds)
            grads = tape.gradient(loss, patch)
            if grads is not None:
                self.optimizer.apply_gradients([(grads, patch)])
            else:
                print('No se encontraron gradientes.')
            
            # Clip del parche para mantenerlo en el rango válido
            patch.assign(tf.clip_by_value(patch, -3, 3))  # Ajusta según el rango de tu imagen preprocesada
            
            print(f'Iteración {i}, Pérdida: {n.mean(loss.numpy())}')
    
        return patch
    
    def __generate_adversarial_image(self) -> Tensor:
        return self.__apply_patch(self.source_image, self.adversarial_patch, self.patch_center)

    def __save_adversarial_patch(self) -> None:
        if self.adversarial_image is None:
            raise ValueError('No se ha generado la imagen adversaria.')

        image = self.get_adversarial_patch()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Asignar la conversión a la variable
        
        # Asegurarse de que el directorio exista

        os.makedirs(os.path.join('adversarial_attacks','generated_patches'), exist_ok=True)
        
        patch_path = os.path.join('adversarial_attacks','generated_patches', f'{self.model.get_name()}_{self.target_class_label}.png')
        
        cv2.imwrite(patch_path, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

    def get_adversarial_image(self) -> Tensor:
        return self.adversarial_image
    
    def get_adversarial_patch(self) -> Tensor:
        return self.adversarial_patch
    
    def get_source_image(self) -> Tensor:
        return self.source_image
    