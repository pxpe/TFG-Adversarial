import customtkinter
from matplotlib.figure import Figure

from PIL import Image
from typing import Union

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AdversarialResult(customtkinter.CTkFrame):
    """
        Clase que implementa un widget personalizado para mostrar los resultados de un ataque adversarial.
    """

    def __init__(self, *args,
                 width: int = 50,
                 height: int = 50,
                 image: Image,
                 descripcion: str,
                 step_size: Union[int, float] = 1,
                 grafico: Figure,
                 font_family: str = "Consolas",
                 prediccion: str = None,
                 **kwargs):
        super().__init__(*args, width=width, height=height, **kwargs)

        self.image = image
        self.descripcion = descripcion
        self.grafico = grafico
        self.step_size = step_size
        self.font_family = font_family
        self.prediccion = prediccion

        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        my_img = customtkinter.CTkImage(light_image=self.image, size=(150, 150))
        self.img_display_label = customtkinter.CTkLabel(self, image=my_img, text=descripcion, compound='top', font=customtkinter.CTkFont(family=self.font_family, size=14, weight="normal"), corner_radius=20)
        self.img_display_label.grid(row=1, column=0, padx=0, pady=(15,5))

        if prediccion is not None:
            self.result_label = customtkinter.CTkLabel(self, text=str(prediccion[0]), font=customtkinter.CTkFont(family=self.font_family, size=16, weight="bold"))
            self.result_label.grid(row=2, column=0, padx=(5,5), pady=(5,15))

            canvas = FigureCanvasTkAgg(self.grafico, master=self)
            canvas.draw()
            canvas.get_tk_widget().grid(row=3, column=0, padx=(5,5), pady=(0,15))


        