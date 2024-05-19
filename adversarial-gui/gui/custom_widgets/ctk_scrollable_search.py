# Autor: José Luis López Ruiz
# Fecha: 19/05/2024
# Descripción: Este script implementa un widget personalizado para hacer una busqueda en scrollbar.



import customtkinter
from matplotlib.figure import Figure

from PIL import Image
from typing import Union

class ScrollableSearch(customtkinter.CTkFrame):
    """
    Clase que implementa un widget personalizado para hacer una búsqueda en un comboBox con campo de input que actualice las opciones del comboBox.
    """

    def __init__(self, parent, entries: dict, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.entries = entries

        # Campo de entrada de búsqueda
        self.search_entry = customtkinter.CTkEntry(self, placeholder_text="Buscar clase...", font=customtkinter.CTkFont(family="Consolas", size=14))
        self.search_entry.pack(side="top", fill="x",padx=10, pady=(10, 5))
        self.search_entry.bind("<KeyRelease>", self.on_search_entry_key_release)

        # Combobox que muestra los resultados de la búsqueda
        self.combo_box = customtkinter.CTkComboBox(self, values=list(self.entries.keys()), dropdown_fg_color='#515151')
        self.combo_box.pack(fill="x", expand=True, padx=10, pady=(10, 5))

    def on_search_entry_key_release(self, event):
        # Filtrar las entradas basadas en el texto de búsqueda
        search_text = self.search_entry.get().lower()
        filtered_keys = [k for k in self.entries.keys() if search_text in k.lower()]
        self.combo_box.configure(values=filtered_keys)
        if filtered_keys:
            self.combo_box.set(filtered_keys[0])
        else:
            self.combo_box.set('')

    def get_selected(self):
        return self.entries[self.combo_box.get()]