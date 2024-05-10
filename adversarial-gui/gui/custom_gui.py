# Autor: José Luis López Ruiz
# Fecha: 04/05/2024
# Descripción: Este script contiene la clase CustomGUI, la cual se encarga de crear la interfaz gráfica de usuario.

import os
from typing import Union


import customtkinter
from customtkinter import filedialog, CTkInputDialog
from .custom_widgets.ctk_yes_no_dialog import CTkYesNoDialog
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from PIL import Image

from model_loader.loader import ModelLoader
from model_loader.model_utils.model_predictions import generate_prediction_graph

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

class AversarialGUI(customtkinter.CTk):

    """
        Clase que representa la interfaz gráfica de usuario de la aplicación.
        Parámetros:
        -    title (str): Título de la ventana.
        -    geometry (str): Tamaño de la ventana.
        -    font_family (str): Fuente de texto a utilizar.
        -    modelLoader (ModelLoader): Instancia del cargador de modelos.
    """


    TEMAS_DISPONIBLES = {
        "Oscuro": "Dark",
        "Claro": "Light"
    }

    def __init__(self, title : str = "Adversarial GUI",geometry : str = "1100x580", font_family : str = "Consolas", modelLoader : ModelLoader = None):
        super().__init__()
        
        self.title_str = title
        self.font_family = font_family
        self.current_image = None
        self.model_loader = modelLoader

        # Configurar la ventana principal
        self.title(title)
        self.geometry(geometry)
        self.resizable(False, False)

        self.grid_columnconfigure(3, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Configurar el sidebar
        self.__instanciar_sidebar_izq()
        # Configurar el contenido
        self.__instanciar_contenido()


    def __instanciar_sidebar_izq(self):
        # Configurar el sidebar
        self.sidebar_frame = customtkinter.CTkFrame(self, width=180, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=1, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text=self.title_str, font=customtkinter.CTkFont(family=self.font_family, size=20, weight="bold"),justify="left")
        self.logo_label.grid(row=0, column=0, padx=20, pady=20)

        self.sidebar_model_label = customtkinter.CTkLabel(self.sidebar_frame, text="Modelo:", anchor='w', font=customtkinter.CTkFont(family=self.font_family, size=12, weight="bold"),justify="left")
        self.sidebar_model_label.grid(row=1, column=0, padx=15, pady=0, sticky="ew")

        self.sidebar_model = customtkinter.CTkComboBox(self.sidebar_frame, font=customtkinter.CTkFont(family=self.font_family, size=12), values=["ResNet50", "SignalModel"], command=lambda model: self.model_loader.switch_model(model))
        self.sidebar_model.grid(row=2, column=0, sticky="ew", padx=15, pady=0)

        self.sidebar_theme_label = customtkinter.CTkLabel(self.sidebar_frame, text="Tema:", anchor='w', font=customtkinter.CTkFont(family=self.font_family, size=12, weight="bold"),justify="left")
        self.sidebar_theme_label.grid(row=5, column=0, sticky="ew",padx=15, pady=0)

        self.sidebar_theme = customtkinter.CTkComboBox(self.sidebar_frame, font=customtkinter.CTkFont(family=self.font_family, size=12), values=["Oscuro", "Claro"], command = lambda tema: self.__cambiar_tema(theme=tema))
        self.sidebar_theme.grid(row=6, column=0, sticky="ew",padx=15, pady=(0, 20))

    def __instanciar_contenido(self):

        # Configurar el contenido principal del GUI

        self.content_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.content_frame.grid(row=0, column=2, rowspan=1, sticky="ew")
        self.content_frame.grid_rowconfigure(2, weight=1)
        self.content_frame.grid_columnconfigure(1, weight=1)

        # Configurar el frame de búsqueda de imagen
        self.img_search_frame = customtkinter.CTkFrame(self.content_frame, width=400, height=400, corner_radius=20)
        self.img_search_frame.grid(row=0, column=0, padx=(20,20), pady=20,sticky="ew")
        self.img_search_frame.grid_rowconfigure(1, weight=1)
        self.img_search_frame.grid_columnconfigure(3, weight=1)

        self.image_label = customtkinter.CTkLabel(self.img_search_frame, text="Imagen:", font=customtkinter.CTkFont(family=self.font_family, size=14, weight="bold"),justify="left")
        self.image_label.grid(row=0, column=1, padx=(20,0), pady=0)

        self.image_label_path = customtkinter.CTkLabel(self.img_search_frame, text=self.current_image, font=customtkinter.CTkFont(family=self.font_family, size=12,slant='italic'),justify="left", wraplength=300, width=350)
        self.image_label_path.grid(row=0, column=2, padx=0, pady=0)

        self.image_find_btn = customtkinter.CTkButton(self.img_search_frame, text="Buscar imagen", font=customtkinter.CTkFont(family=self.font_family, size=12), command=self.__buscar_imagen)
        self.image_find_btn.grid(row=0, column=3, padx=0, pady=0)

 
    def __instanciar_widgets_imagen(self):
       
        # Configurar el frame de visualización de imagen
        self.img_display_frame = customtkinter.CTkFrame(self.content_frame, corner_radius=20)
        self.img_display_frame.grid(row=1, column=0, padx=0, pady=0, sticky="ew")
        self.img_display_frame.grid_rowconfigure(1, weight=0)
        self.img_display_frame.grid_columnconfigure(1, weight=0)

        my_img = customtkinter.CTkImage(light_image=Image.open(str(self.current_image)), size=(400, 400))
        self.img_display_label = customtkinter.CTkLabel(self.img_display_frame, image=my_img, text="Previsualización de la imágen", compound='top', font=customtkinter.CTkFont(family=self.font_family, size=14, weight="normal"), corner_radius=20)
        self.img_display_label.grid(row=1, column=0, padx=0, pady=(20,20))

        # Configurar el frame de botones para la imagen
        self.img_btn_frame = customtkinter.CTkFrame(self.img_display_frame, corner_radius=20)
        self.img_btn_frame.grid(row=1, column=1, rowspan=4, padx=10,sticky="ew")
        self.img_btn_frame.grid_rowconfigure(3, weight=1)

        self.img_btn_analizar = customtkinter.CTkButton(self.img_btn_frame, text="Obtener predicción", font=customtkinter.CTkFont(family=self.font_family, size=12), command=self.__realizar_prediccion)
        self.img_btn_analizar.grid(row=0, column=0, padx=0, pady=0)




    def __buscar_imagen(self):

        """
            Método que permite al usuario seleccionar una imagen de su sistema de archivos.
        """

        filename =  filedialog.askopenfilename(parent=self,title="Seleccionar imagen", initialdir= os.curdir, filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp")], defaultextension=".jpg", multiple=False)
        if filename == "": return

        self.current_image = filename
        messagebox.showinfo("Información", "Imagen cargada correctamente")
        imgDir = filename.split("/")[-2:]
        self.image_label_path.configure(text=f".../{'/'.join(imgDir)}")
        self.__mostrar_imagen()

    def __mostrar_imagen(self):
        """
            Método que muestra la imagen seleccionada por el usuario en la interfaz gráfica.
        """

        self.__instanciar_widgets_imagen()
    
    def __cambiar_tema(self, theme : str) -> None:
        """
            Método que cambia el tema de la interfaz gráfica.
        """

        try:
            customtkinter.set_appearance_mode(self.TEMAS_DISPONIBLES[theme])
        except KeyError:
            print(f"El tema '{theme}' no está disponible. Los temas disponibles son: {', '.join(self.TEMAS_DISPONIBLES.keys())}")
            

    def __realizar_prediccion(self):
        """
            Método que realiza una predicción sobre la imagen cargada.
        """

        prediccion = self.model_loader.predict(self.current_image)
        if prediccion:
            print(prediccion[1])

            self.prediccion_label = customtkinter.CTkLabel(self.img_btn_frame, text=f'Predicción: {str(prediccion[0])}', font=customtkinter.CTkFont(family=self.font_family, size=14, weight="bold"),justify="left")
            self.prediccion_label.grid(row=1, column=0, padx=(5,5), pady=(0,15))

            # Generar gráfico de barras con las últimas cuatro predicciones
            fig, ax = generate_prediction_graph(prediccion[1])

            canvas = FigureCanvasTkAgg(fig, master=self.img_btn_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=2, column=0, padx=(5,5), pady=(0,15))

            
        else:
            messagebox.showerror("Error", "No se ha podido realizar la predicción. Asegúrese de que el modelo esté cargado correctamente.")






