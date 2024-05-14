import customtkinter
from matplotlib.figure import Figure

from PIL import Image
from typing import Union

class AdversarialResult(customtkinter.CTkFrame):
    """
        Clase que implementa un widget personalizado para mostrar los resultados de un ataque adversarial.
    """

    def __init__(self, *args,
                 width: int = 100,
                 height: int = 32,
                 image: Image,
                 descripcion: str,
                 step_size: Union[int, float] = 1,
                 grafico: Figure,
                 font_family: str = "Consolas",
                 **kwargs):
        super().__init__(*args, width=width, height=height, **kwargs)

        self.image = image
        self.descripcion = descripcion
        self.grafico = grafico
        self.step_size = step_size
        self.font_family = font_family

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        my_img = customtkinter.CTkImage(light_image=self.image, size=(300, 300))
        self.img_display_label = customtkinter.CTkLabel(self, image=my_img, text=descripcion, compound='top', font=customtkinter.CTkFont(family=self.font_family, size=14, weight="normal"), corner_radius=20)
        self.img_display_label.grid(row=1, column=0, padx=0, pady=(20,20))

        self.grafico_canvas = customtkinter.CTkCanvas(self, width=300, height=300)
        self.grafico_canvas.grid(row=2, column=0, padx=0, pady=0)
        self.grafico_canvas.add_figure(self.grafico)



        