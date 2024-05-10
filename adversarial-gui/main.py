# Autor: José Luis López Ruiz
# Fecha: 04/05/2024
# Descripción: Este script ejecuta la aplicación de interfaz gráfica de usuario.


from gui.custom_gui import AversarialGUI
from model_loader.loader import ModelLoader

if __name__ == "__main__":
    """
        Función principal que ejecuta la aplicación de interfaz gráfica de usuario.
    """
    modelLoader = ModelLoader()
    app = AversarialGUI(modelLoader=modelLoader)
    app.mainloop()