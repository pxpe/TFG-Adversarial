import numpy as np

# Divide imagen de tamaño AxA en N imagenes de tamaño BxB
def divide_image(image, a, b):
    images = []
    for i in range(0, a, b):
        for j in range(0, a, b):
            images.append(image[i:i+b, j:j+b])
    return images

# Reconstruye una imagen de tamaño AxA a partir de N imagenes de tamaño BxB
def reconstruct_image(images, a, b):
    image = np.zeros((a, a, 3))
    k = 0
    for i in range(0, a, b):
        for j in range(0, a, b):
            image[i:i+b, j:j+b] = images[k]
            k += 1
    return image