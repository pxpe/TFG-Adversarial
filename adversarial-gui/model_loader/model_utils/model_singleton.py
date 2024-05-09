# Autor: José Luis López Ruiz
# Fecha: 08/05/2024
# Descripción: Este script define el decorador Singleton, el cual se encarga de garantizar que una clase solo tenga una instancia en ejecución.


def Singleton(cls):
    """
        Decorador Singleton que garantiza que una clase solo tenga una instancia en ejecución.
    """

    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance