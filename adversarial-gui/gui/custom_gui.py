# Autor: José Luis López Ruiz
# Fecha: 04/05/2024
# Descripción: Este script contiene la clase CustomGUI, la cual se encarga de crear la interfaz gráfica de usuario.

import os
from typing import Union, Literal
from time import time

import customtkinter
from customtkinter import filedialog, CTkInputDialog
from .custom_widgets.ctk_adversarial_result import AdversarialResult
from .custom_widgets.ctk_scrollable_search import ScrollableSearch
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt

from PIL import Image
import numpy as np
import tensorflow as tf

from model_loader.loader import ModelLoader
from adversarial_attacks.FGSM import FGSMAttack
from adversarial_attacks.adversarial_patch import AdversarialPatch
from adversarial_defenses.adversarial_purification.Cifar10DenoiserAutoencoder import Cifar10DenoiserAutoEncoder
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

    MODELOS_DISPONIBLES = [ "ResNet50 V2", "SignalModel", "AdvTrainedSignalModel", "MobileNet V2", "VGG19"]

    ATAQUES_DISPONIBLES = ["N/A","FGSM","Parche Adversario"]

    DEFENSAS_DISPONIBLES = ["N/A", "Purificación Adversaria"]

    def __init__(self, title : str = "Adversarial GUI",geometry : str = "1300x680", font_family : str = "Consolas", modelLoader : ModelLoader = None):
        super().__init__()
        
        self.title_str = title
        self.font_family = font_family
        self.current_image = None
        self.model_loader = modelLoader
        self.attack = None
        self.defense = None

        # Configurar la ventana principal
        self.title(title)
        self.geometry(geometry)
        self.resizable(True, True)

        self.grid_columnconfigure(3, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Configurar el sidebar
        self.__instanciar_sidebar_izq()
        # Configurar el contenido
        self.__instanciar_contenido()


    def __instanciar_sidebar_izq(self):
        # Configurar el sidebar
        self.sidebar_frame = customtkinter.CTkFrame(self, width=180, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=1, rowspan=10, sticky="nsew")
        
        # Configurar las filas intermedias con peso positivo
        self.sidebar_frame.grid_rowconfigure(8, weight=1)  # Espacio expansivo
        self.sidebar_frame.grid_rowconfigure(9, weight=0)  # Fila para el label de tema
        self.sidebar_frame.grid_rowconfigure(10, weight=0)  # Fila para el combo box de tema

        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text=self.title_str, font=customtkinter.CTkFont(family=self.font_family, size=20, weight="bold"), justify="left")
        self.logo_label.grid(row=0, column=0, padx=20, pady=20)

        self.sidebar_model_label = customtkinter.CTkLabel(self.sidebar_frame, text="Modelo:", anchor='w', font=customtkinter.CTkFont(family=self.font_family, size=12, weight="bold"), justify="left")
        self.sidebar_model_label.grid(row=1, column=0, padx=15, pady=0, sticky="ew")

        self.sidebar_model = customtkinter.CTkComboBox(self.sidebar_frame, font=customtkinter.CTkFont(family=self.font_family, size=12), values=self.MODELOS_DISPONIBLES, command=lambda model: self.__cambiar_modelo(model))
        self.sidebar_model.grid(row=2, column=0, sticky="ew", padx=15, pady=0)

        self.sidebarl_attack_label = customtkinter.CTkLabel(self.sidebar_frame, text="Ataque:", anchor='w', font=customtkinter.CTkFont(family=self.font_family, size=12, weight="bold"), justify="left")
        self.sidebarl_attack_label.grid(row=3, column=0, sticky="ew", padx=15, pady=(5, 0))

        self.sidebar_attack = customtkinter.CTkComboBox(self.sidebar_frame, font=customtkinter.CTkFont(family=self.font_family, size=12), values=self.ATAQUES_DISPONIBLES, command=lambda attack: self.__cambiar_ataque(attack))
        self.sidebar_attack.grid(row=4, column=0, sticky="ew", padx=15, pady=0)

        self.sidebar_defense_label = customtkinter.CTkLabel(self.sidebar_frame, text="Defensa:", anchor='w', font=customtkinter.CTkFont(family=self.font_family, size=12, weight="bold"), justify="left")
        self.sidebar_defense_label.grid(row=6, column=0, sticky="ew", padx=15, pady=(5, 0))

        self.sidebar_defense = customtkinter.CTkComboBox(self.sidebar_frame, font=customtkinter.CTkFont(family=self.font_family, size=12), values=self.DEFENSAS_DISPONIBLES, command=lambda defense: self.__cambiar_defensa(defense))
        self.sidebar_defense.grid(row=7, column=0, sticky="ew", padx=15, pady=0)

        self.sidebar_theme_label = customtkinter.CTkLabel(self.sidebar_frame, text="Tema:", anchor='w', font=customtkinter.CTkFont(family=self.font_family, size=12, weight="bold"), justify="left")
        self.sidebar_theme_label.grid(row=9, column=0, sticky="ew", padx=15, pady=0)

        self.sidebar_theme = customtkinter.CTkComboBox(self.sidebar_frame, font=customtkinter.CTkFont(family=self.font_family, size=12), values=["Oscuro", "Claro"], command=lambda tema: self.__cambiar_tema(theme=tema))
        self.sidebar_theme.grid(row=10, column=0, sticky="ew", padx=15, pady=(0, 20))

    def __instanciar_contenido(self):

        # Configurar el contenido principal del GUI

        self.content_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.content_frame.grid(row=0, column=2, rowspan=1,columnspan=4, sticky="ew")
        self.content_frame.grid_rowconfigure(6, weight=1)
        self.content_frame.grid_columnconfigure(1, weight=1)

        # Configurar el frame de búsqueda de imagen
        self.img_search_frame = customtkinter.CTkFrame(self.content_frame, width=400, height=400, corner_radius=20)
        self.img_search_frame.grid(row=0, column=0, padx=(20,20), pady=20)
        self.img_search_frame.grid_rowconfigure(1, weight=1)
        self.img_search_frame.grid_columnconfigure(6, weight=4)

        self.image_label = customtkinter.CTkLabel(self.img_search_frame, text="Imagen:", font=customtkinter.CTkFont(family=self.font_family, size=14, weight="bold"),justify="left")
        self.image_label.grid(row=0, column=1, padx=(20,0), pady=0)

        self.image_label_path = customtkinter.CTkLabel(self.img_search_frame, text=self.current_image, font=customtkinter.CTkFont(family=self.font_family, size=12,slant='italic'),justify="left", wraplength=300, width=350)
        self.image_label_path.grid(row=0, column=2, padx=0, pady=0)

        self.image_find_btn = customtkinter.CTkButton(self.img_search_frame, text="Buscar imagen", font=customtkinter.CTkFont(family=self.font_family, size=12), command=self.__buscar_imagen)
        self.image_find_btn.grid(row=0, column=3, padx=0, pady=0)

 
    def __instanciar_widgets_imagen(self):
        self.__limpiar_contenido()
        
        # Configurar el frame de visualización de imagen
        self.img_display_frame = customtkinter.CTkFrame(self.content_frame, corner_radius=20)
        self.img_display_frame.grid(row=1, column=0, padx=20, pady=(0,50), sticky="ew")
        self.img_display_frame.grid_rowconfigure(1, weight=0)
        self.img_display_frame.grid_columnconfigure(1, weight=0)

        my_img = customtkinter.CTkImage(light_image=Image.open(str(self.current_image)), size=(350, 350))
        self.img_display_label = customtkinter.CTkLabel(self.img_display_frame, image=my_img, text="Previsualización de la imágen", compound='top', font=customtkinter.CTkFont(family=self.font_family, size=14, weight="normal"), corner_radius=20)
        self.img_display_label.grid(row=1, column=0, padx=0, pady=(20,20))

        # Configurar el frame de botones para la imagen
        self.img_btn_frame = customtkinter.CTkFrame(self.img_display_frame, corner_radius=20)
        self.img_btn_frame.grid(row=1, column=1, rowspan=4, padx=10, pady=10,sticky="ew")
        self.img_btn_frame.grid_rowconfigure(4, weight=1)

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

    def __limpiar_attack_params(self):
        try:
            self.sidebar_attack_params.destroy()
        except AttributeError:
            pass

    def __limpiar_contenido(self):
        try:
            for widget in self.content_frame.winfo_children():
                if widget != self.img_search_frame:
                    widget.destroy()
        except AttributeError:
            pass

        

    def __mostrar_fgsm_params(self):

        self.sidebar_attack_params = customtkinter.CTkFrame(self.sidebar_frame, corner_radius=10)
        self.sidebar_attack_params.grid(row=5, column=0, sticky="ew",padx=10, pady=15)
        self.sidebar_attack_params.grid_columnconfigure(1, weight=1)

        self.fgsm_epsiplon_label = customtkinter.CTkLabel(self.sidebar_attack_params, text="Epsilon:", anchor='w', font=customtkinter.CTkFont(family=self.font_family, size=12, weight="bold"),justify="left")
        self.fgsm_epsiplon_label.grid(row=0, column=0, sticky="ew",padx=15, pady=15)

        self.fgsm_epsilon = customtkinter.CTkEntry(self.sidebar_attack_params, font=customtkinter.CTkFont(family=self.font_family, size=12), width=10, placeholder_text="0.1")
        self.fgsm_epsilon.grid(row=0, column=1, sticky="ew",padx=15, pady=15)

    def __mostrar_parche_params(self):

        self.sidebar_attack_params = customtkinter.CTkFrame(self.sidebar_frame, corner_radius=10)
        self.sidebar_attack_params.grid(row=5, column=0, sticky="ew",padx=10, pady=15)
        self.sidebar_attack_params.grid_rowconfigure(1, weight=1)

        self.patch_class_label = customtkinter.CTkLabel(self.sidebar_attack_params, text="Clase Objetivo:", anchor='w', font=customtkinter.CTkFont(family=self.font_family, size=12, weight="bold"),justify="left")
        self.patch_class_label.grid(row=0, column=0, sticky="ew",padx=15, pady=(5,0))

        self.patch_target_class = ScrollableSearch(self.sidebar_attack_params, entries=self.model_loader.get_classes())
        self.patch_target_class.grid(row=1, column=0, sticky="ew",padx=15, pady=(5,15))


    def __mostrar_imagen(self):
        """
            Método que muestra la imagen seleccionada por el usuario en la interfaz gráfica.
        """

        self.__instanciar_widgets_imagen()

    def __cambiar_ataque(self, attack : str) -> None:
        """
            Método que cambia el ataque a realizar sobre la imagen cargada.
        """
        self.__limpiar_attack_params()
 

        if attack == "N/A":
            self.attack = None
        else:
            self.attack = attack
            if attack == "FGSM":
                self.__mostrar_fgsm_params()
            elif attack == "Parche Adversario":
                self.__mostrar_parche_params()

    def __cambiar_defensa(self, defense : str) -> None:
        """
            Método que cambia la defensa a aplicar sobre la imagen cargada.
        """
        self.defense = None if defense == "N/A" else defense
    
    def __cambiar_modelo(self, model : str) -> None:
        """
            Método que cambia el modelo de red neuronal a utilizar.
        """
        self.model_loader.switch_model(model)
        self.__limpiar_attack_params()
        self.attack = None
        self.sidebar_attack.set("N/A")
        messagebox.showinfo("Información", f"Modelo cambiado a {model}")


    def __cambiar_tema(self, theme : str) -> None:
        """
            Método que cambia el tema de la interfaz gráfica.
        """

        try:
            customtkinter.set_appearance_mode(self.TEMAS_DISPONIBLES[theme])
        except KeyError:
            print(f"El tema '{theme}' no está disponible. Los temas disponibles son: {', '.join(self.TEMAS_DISPONIBLES.keys())}")
   

    def __mostrarResultadosAdversariosPurificados(self, img_original, img_perturbacion, img_adversaria,img_adversaria_purificada, prediccion_real, prediccion_adversaria, prediccion_purificada , desc1, desc2, desc3,descPurificada, fig1, fig2, fig3):
        # Configurar el frame de resultados
        self.__limpiar_contenido()

        self.result_frame = customtkinter.CTkFrame(self.content_frame,width=400, height=400, corner_radius=20)
        self.result_frame.grid(row=2, column=0, padx=15, pady=15, sticky="ew")
        self.result_frame.grid_rowconfigure(1, weight=0)
        self.result_frame.grid_columnconfigure(4, weight=1)

        

        self.real_result = AdversarialResult(self.result_frame, width=100, height=130, image=img_original, descripcion=desc1, prediccion=prediccion_real, step_size=0, grafico=fig1, font_family=self.font_family)
        self.real_result.grid(row=0, column=0, padx=5, pady=15)

        self.perturbacion_result = AdversarialResult(self.result_frame, width=50, height=50, image=img_perturbacion, descripcion=desc2, grafico=None, font_family=self.font_family)
        self.perturbacion_result.grid(row=1, column=1, padx=5, pady=5)

        self.adversarial_result = AdversarialResult(self.result_frame, width=100, height=130, image=img_adversaria,prediccion=prediccion_adversaria, descripcion=desc3, grafico=fig2, font_family=self.font_family)
        self.adversarial_result.grid(row=0, column=1, padx=5, pady=15)

        self.adversarial_purified_result = AdversarialResult(self.result_frame, width=100, height=130, image=img_adversaria_purificada, prediccion=prediccion_purificada,descripcion=descPurificada, grafico=fig3, font_family=self.font_family)
        self.adversarial_purified_result.grid(row=0, column=2, padx=5, pady=5)

        self.result_frame.grid_rowconfigure(1, weight=0)

    def __mostrarResultadosAdversarios(self, img_original, img_perturbacion, img_adversaria, prediccion_real, prediccion_adversaria, desc1, desc2, desc3, fig1, fig2):
        # Configurar el frame de resultados
        self.__limpiar_contenido()

        self.result_frame = customtkinter.CTkFrame(self.content_frame,width=400, height=400, corner_radius=20)
        self.result_frame.grid(row=2, column=0, padx=15, pady=15, sticky="ew")
        self.result_frame.grid_rowconfigure(1, weight=0)
        self.result_frame.grid_columnconfigure(3, weight=1)


        self.real_result = AdversarialResult(self.result_frame, width=150, height=180, image=img_original, descripcion=desc1, prediccion=prediccion_real, step_size=0, grafico=fig1, font_family=self.font_family)
        self.real_result.grid(row=0, column=0, padx=15, pady=15)

        self.perturbacion_result = AdversarialResult(self.result_frame, width=150, height=180, image=img_perturbacion, descripcion=desc2, grafico=None, font_family=self.font_family)
        self.perturbacion_result.grid(row=0, column=1, padx=15, pady=15)

        self.adversarial_result = AdversarialResult(self.result_frame, width=150, height=180, image=img_adversaria,prediccion=prediccion_adversaria, descripcion=desc3, grafico=fig2, font_family=self.font_family)
        self.adversarial_result.grid(row=0, column=2, padx=15, pady=15)

    def __predecir_fgsm(self):
        epsilon = self.fgsm_epsilon.get().replace(',', '.').strip()

        try:
            epsilon = float(epsilon)
            if epsilon <= 0 or epsilon > 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "El valor de epsilon es incorrecto. " + epsilon)
            return
        
        
        prediccion_real = self.model_loader.predict(self.current_image)
        fig1, _ = generate_prediction_graph(prediccion_real[1])

        try:
            label = self.model_loader.get_label(prediccion_real[0].predicted_class)
        except KeyError:
            messagebox.showerror("Error", f"No se ha podido obtener la etiqueta de la predicción real ( {str(prediccion_real[0].predicted_class)} )")
            return

        label = self.model_loader.get_label(prediccion_real[0].predicted_class)

        fgsm = FGSMAttack(self.current_image, epsilon = epsilon,input_label = label, model = self.model_loader)

        imagen_original = fgsm.get_source_image()
        perturbacion = fgsm.get_adversarial_pattern()
        imagen_adversaria = fgsm.get_adversarial_image()        

        prediccion_adversaria = self.model_loader.predict(imagen_adversaria)
        fig2, _ = generate_prediction_graph(prediccion_adversaria[1])
   
        img_original = self.model_loader.normalize_image(imagen_original)
        
        img_perturbacion = self.model_loader.normalize_image(perturbacion)

        img_adversaria = self.model_loader.normalize_image(imagen_adversaria)

        if self.defense is None:
            self.__mostrarResultadosAdversarios(img_original, img_perturbacion, img_adversaria, prediccion_real, prediccion_adversaria, "Imagen original", "Perturbación", f"Imagen adversaria con episilon = {epsilon}" , fig1, fig2)
        
        elif self.defense == "Purificación Adversaria":
            try:
                purificador = Cifar10DenoiserAutoEncoder(imagen_adversaria[0])
                img_adversaria_purificada = purificador.get_purified_image()
                img_adversaria_purificada_reshape = self.model_loader.reshape_image(img_adversaria_purificada)
                
                predictions_purified_image = self.model_loader.predict(img_adversaria_purificada_reshape)
                fig3, _ = generate_prediction_graph(predictions_purified_image[1])

                # Convert the numpy array to a PIL image
                pil_image = Image.fromarray((img_adversaria_purificada * 255.0).astype("uint8"))

                self.__mostrarResultadosAdversariosPurificados(img_original, img_perturbacion, img_adversaria, pil_image, prediccion_real, prediccion_adversaria,predictions_purified_image, "Imagen original", "Perturbación", "Imagen adversaria", f'Imagen adversaria purificada', fig1, fig2, fig3)

            except Exception as e:
                print(e)
                messagebox.showerror("Error", "No se ha podido aplicar la defensa de Purificación Adversaria.")        



    def __predecir_parche(self):
        target_class = self.patch_target_class.get_selected()
        print(target_class)
        if target_class is None:
            messagebox.showerror("Error", "No se ha seleccionado una clase objetivo.")
            return
        
        try:
            target_class = int(target_class)
        except ValueError:
            messagebox.showerror("Error", "La clase objetivo no es válida.")
            return
        
        prediccion_real = self.model_loader.predict(self.current_image)
        fig1, _ = generate_prediction_graph(prediccion_real[1])

        try:
            patch = AdversarialPatch(self.current_image, target_class, self.model_loader)
        except Exception as e:
            print(e)
            messagebox.showerror("Error", f"No se ha podido generar el parche adversario. Vuelve a intentarlo o cambia de clase objetivo.")
            return

        original_img = patch.get_source_image()
        parche_adversario = patch.get_adversarial_patch()
        adversarial_image = patch.get_adversarial_image()

        adversarial_prediction = self.model_loader.predict(adversarial_image)

        fig2, _ = generate_prediction_graph(adversarial_prediction[1])

        img_original = self.model_loader.normalize_image(original_img)
        
        img_perturbacion = self.model_loader.normalize_patch(parche_adversario)

        img_adversaria = self.model_loader.normalize_image(adversarial_image)

        self.__mostrarResultadosAdversarios(img_original, img_perturbacion, img_adversaria, prediccion_real, adversarial_prediction, "Imagen original", "Parche adversario", "Imagen adversaria", fig1, fig2)






    def __predecir_normal(self):
        time_start = time()
        prediccion = self.model_loader.predict(self.current_image)
        time_end = time()
        if prediccion:

            total_time = round(time_end - time_start, 2)
            
            self.img_btn_analizar.destroy()
            self.prediccion_label = customtkinter.CTkLabel(self.img_btn_frame, text=f'Predicción: {str(prediccion[0])}', font=customtkinter.CTkFont(family=self.font_family, size=14, weight="bold"),justify="left")
            self.prediccion_label.grid(row=1, column=0, padx=(5,5), pady=(0,15))

            # Generar gráfico de barras con las últimas cuatro predicciones
            fig, ax = generate_prediction_graph(prediccion[1])

            canvas = FigureCanvasTkAgg(fig, master=self.img_btn_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=2, column=0, padx=(5,5), pady=(0,15))

            self.time_label = customtkinter.CTkLabel(self.img_btn_frame, text=f'Tiempo de predicción: {total_time} segundos', font=customtkinter.CTkFont(family=self.font_family, size=14, weight="bold"),justify="left")
            self.time_label.grid(row=3, column=0, padx=(5,5), pady=(0,15))
        else:
            messagebox.showerror("Error", "No se ha podido realizar la predicción. Asegúrese de que el modelo esté cargado correctamente.")


    def __realizar_prediccion(self):
        """
            Método que realiza una predicción sobre la imagen cargada.
        """
        if self.current_image is None:
            messagebox.showerror("Error", "No se ha cargado ninguna imagen.")
            return

        if self.attack is None:
            self.__predecir_normal()
        else:
            if self.attack == "FGSM":
                self.__predecir_fgsm()
            elif self.attack == "Parche Adversario":
                self.__predecir_parche()






