import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import numpy as np
import pandas as pd
from scipy import stats
import os

# =============================================================================
# FUNCIONES ESTADÍSTICAS MODULARES (NUEVAS)
# =============================================================================

def validar_y_convertir_datos(datos_str):
    """Valida datos de entrada y los convierte a numpy array."""
    try:
        if not datos_str.strip():
            raise ValueError("No se ingresaron datos")
        
        datos_str = datos_str.replace('\n', ',').replace(';', ',').strip()
        items_datos = [item.strip() for item in datos_str.split(',') if item.strip()]
        datos = [float(item) for item in items_datos if item]
        
        if not datos:
            raise ValueError("No hay valores numéricos válidos")
            
        return np.array(datos)
    except ValueError as e:
        messagebox.showerror("Error", f"Datos inválidos: {str(e)}")
        return None

def calcular_intervalo_z(datos, confianza):
    """Calcula intervalo de confianza Z."""
    n = len(datos)
    media = np.mean(datos)
    desv_std = np.std(datos, ddof=1)
    error_std = desv_std / np.sqrt(n)
    z_critico = stats.norm.ppf((1 + confianza) / 2)
    margen = z_critico * error_std
    return {
        "inferior": media - margen,
        "superior": media + margen,
        "estadisticas": {
            "n": n, "media": media, "desv_std": desv_std, "error_std": error_std,
            "critico": z_critico, "margen": margen, "prueba": "Z"
        }
    }

def calcular_intervalo_t(datos, confianza):
    """Calcula intervalo de confianza t."""
    n = len(datos)
    media = np.mean(datos)
    desv_std = np.std(datos, ddof=1)
    error_std = desv_std / np.sqrt(n)
    t_critico = stats.t.ppf((1 + confianza) / 2, df=n-1)
    margen = t_critico * error_std
    return {
        "inferior": media - margen,
        "superior": media + margen,
        "estadisticas": {
            "n": n, "media": media, "desv_std": desv_std, "error_std": error_std,
            "critico": t_critico, "margen": margen, "gl": n-1, "prueba": "t"
        }
    }

def realizar_prueba_z(datos, valor_nulo, alpha, direccion):
    """Realiza prueba Z de hipótesis."""
    n = len(datos)
    media = np.mean(datos)
    desv_std = np.std(datos, ddof=1)
    error_std = desv_std / np.sqrt(n)
    estadistico = (media - valor_nulo) / error_std
    
    if "dos colas" in direccion:
        valor_p = 2 * (1 - stats.norm.cdf(abs(estadistico)))
        hipotesis_alt = f"μ ≠ {valor_nulo}"
    elif "cola izquierda" in direccion:
        valor_p = stats.norm.cdf(estadistico)
        hipotesis_alt = f"μ < {valor_nulo}"
    else:
        valor_p = 1 - stats.norm.cdf(estadistico)
        hipotesis_alt = f"μ > {valor_nulo}"
    
    return {
        "estadistico": estadistico, "valor_p": valor_p,
        "estadisticas": {
            "n": n, "media": media, "desv_std": desv_std, "error_std": error_std,
            "hipotesis_alt": hipotesis_alt, "prueba": "Z"
        }
    }

def realizar_prueba_t(datos, valor_nulo, alpha, direccion):
    """Realiza prueba t de hipótesis."""
    n = len(datos)
    gl = n - 1
    media = np.mean(datos)
    desv_std = np.std(datos, ddof=1)
    error_std = desv_std / np.sqrt(n)
    estadistico = (media - valor_nulo) / error_std
    
    if "dos colas" in direccion:
        valor_p = 2 * (1 - stats.t.cdf(abs(estadistico), df=gl))
        hipotesis_alt = f"μ ≠ {valor_nulo}"
    elif "cola izquierda" in direccion:
        valor_p = stats.t.cdf(estadistico, df=gl)
        hipotesis_alt = f"μ < {valor_nulo}"
    else:
        valor_p = 1 - stats.t.cdf(estadistico, df=gl)
        hipotesis_alt = f"μ > {valor_nulo}"
    
    return {
        "estadistico": estadistico, "valor_p": valor_p,
        "estadisticas": {
            "n": n, "media": media, "desv_std": desv_std, "error_std": error_std,
            "hipotesis_alt": hipotesis_alt, "gl": gl, "prueba": "t"
        }
    }

# =============================================================================
# CLASE PRINCIPAL (GUI - CÓDIGO ORIGINAL COMPLETO)
# =============================================================================

class AplicacionEstadistica:
    def __init__(self, raiz):
        self.raiz = raiz
        self.raiz.title("Análisis Estadístico")
        self.raiz.geometry("900x600")
        self.raiz.minsize(700, 500)
        
        self.colores = {
            "primario": "#000000",
            "secundario": "#000080",
            "acento": "#3498db",
            "hover": "#2980b9",
            "claro": "#ecf0f1",
            "texto": "#000000",
            "texto_claro": "#ecf0f1",
            "texto_negro": "#000000"
        }
        
        self.configurar_estilos()
        self.crear_interfaz()
        
    def configurar_estilos(self):
        self.estilo = ttk.Style()
        
        self.estilo.configure("TNotebook", background=self.colores["claro"])
        self.estilo.configure("TNotebook.Tab", 
                            background=self.colores["claro"], 
                            foreground=self.colores["texto"], 
                            padding=[12, 8],
                            font=("Arial", 10, "bold"))
        self.estilo.map("TNotebook.Tab", 
                      background=[("selected", self.colores["acento"])],
                      foreground=[("selected", self.colores["texto_claro"])])
        
        self.estilo.configure("Panel.TFrame", background=self.colores["claro"], relief="raised")
        
        self.estilo.configure("Modern.TButton", background=self.colores["acento"], 
                             foreground=self.colores["texto_negro"], padding=10,
                             font=("Arial", 10, "bold"))
        self.estilo.map("Modern.TButton", 
                       background=[("active", self.colores["hover"])],
                       relief=[("pressed", "sunken")])
        
        self.estilo.configure("Modern.TEntry", padding=8, relief="flat")
        
        self.estilo.configure("Modern.TLabel", background=self.colores["claro"], 
                             foreground=self.colores["texto"],
                             font=("Arial", 10))
        
        self.estilo.configure("Title.TLabel", background=self.colores["claro"], 
                             foreground=self.colores["primario"],
                             font=("Arial", 12, "bold"))
                             
    def crear_interfaz(self):
        self.contenedor_principal = ttk.Frame(self.raiz)
        self.contenedor_principal.pack(fill="both", expand=True)
        
        self.crear_barra_lateral()
        
        self.contenido = ttk.Frame(self.contenedor_principal, style="Panel.TFrame")
        self.contenido.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        self.crear_pestanas()
        
    def crear_barra_lateral(self):
        self.barra_lateral = tk.Frame(self.contenedor_principal, width=200, bg=self.colores["primario"])
        self.barra_lateral.pack(side="left", fill="y", padx=0, pady=0)
        self.barra_lateral.pack_propagate(False)
        
        marco_logo = tk.Frame(self.barra_lateral, bg=self.colores["primario"], height=120)
        marco_logo.pack(fill="x")
        
        etiqueta_titulo = tk.Label(marco_logo, text="ANÁLISIS\nESTADÍSTICO", font=("Arial", 14, "bold"),
                              fg=self.colores["texto_claro"], bg=self.colores["primario"])
        etiqueta_titulo.pack(fill="both", expand=True, pady=20)
        
        separador = ttk.Separator(self.barra_lateral, orient="horizontal")
        separador.pack(fill="x", padx=20)
        
        opciones_menu = [
            {"text": "Intervalos de Confianza", "command": lambda: self.mostrar_pestana(0)},
            {"text": "Pruebas de Media", "command": lambda: self.mostrar_pestana(1)},
            {"text": "Guardar Resultados", "command": self.guardar_resultados},
            {"text": "Ayuda" , "command": self.mostrar_ayuda}
        ]
        
        contenedor_botones = tk.Frame(self.barra_lateral, bg=self.colores["primario"])
        contenedor_botones.pack(fill="both", expand=True, pady=10)
        
        for opcion in opciones_menu:
            self.crear_boton_menu(contenedor_botones, opcion)
            
        pie_pagina = tk.Frame(self.barra_lateral, bg=self.colores["primario"], height=40)
        pie_pagina.pack(fill="x", side="bottom")
        
        etiqueta_version = tk.Label(pie_pagina, text="v1.0.2", font=("Arial", 8),
                                fg=self.colores["texto_claro"], bg=self.colores["primario"])
        etiqueta_version.pack(side="right", padx=10, pady=10)
        
        self.datos = None
        self.texto_resultados = ""
            
    def crear_boton_menu(self, padre, opcion):
        marco_boton = tk.Frame(padre, bg=self.colores["primario"])
        marco_boton.pack(fill="x", pady=2)
        
        def al_entrar(e):
            marco_boton.config(bg=self.colores["secundario"])
            etiqueta_texto.config(bg=self.colores["secundario"])
            
        def al_salir(e):
            marco_boton.config(bg=self.colores["primario"])
            etiqueta_texto.config(bg=self.colores["primario"])
            
        marco_boton.bind("<Enter>", al_entrar)
        marco_boton.bind("<Leave>", al_salir)
        marco_boton.bind("<Button-1>", lambda e: opcion["command"]())
        
        
        etiqueta_texto = tk.Label(marco_boton, text=opcion["text"], font=("Arial", 10),
                             fg=self.colores["texto_claro"], bg=self.colores["primario"],
                             anchor="w")
        etiqueta_texto.pack(side="left", fill="x", padx=5, pady=8)
        etiqueta_texto.bind("<Enter>", al_entrar)
        etiqueta_texto.bind("<Leave>", al_salir)
        etiqueta_texto.bind("<Button-1>", lambda e: opcion["command"]())
        
    def crear_pestanas(self):
        self.cuaderno = ttk.Notebook(self.contenido)
        self.cuaderno.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.pestana1 = ttk.Frame(self.cuaderno, style="Panel.TFrame")
        self.cuaderno.add(self.pestana1, text="Intervalos de Confianza")
        self.crear_pestana_intervalo_confianza()
        
        self.pestana2 = ttk.Frame(self.cuaderno, style="Panel.TFrame")
        self.cuaderno.add(self.pestana2, text="Pruebas de Media")
        self.crear_pestana_prueba_media()
        
    def crear_pestana_intervalo_confianza(self):
        marco_principal = ttk.Frame(self.pestana1)
        marco_principal.pack(fill="both", expand=True, padx=20, pady=10)
        
        panel_izquierdo = ttk.LabelFrame(marco_principal, text="Datos de Entrada", padding=15)
        panel_izquierdo.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
        
        etiqueta_datos = ttk.Label(panel_izquierdo, text="Datos de la muestra (separados por comas):", style="Modern.TLabel")
        etiqueta_datos.pack(anchor="w", pady=(5, 2))
        
        self.marco_datos_ic = ttk.Frame(panel_izquierdo)
        self.marco_datos_ic.pack(fill="x", pady=(0, 10))
        
        self.entrada_datos_ic = scrolledtext.ScrolledText(self.marco_datos_ic, height=5, width=30)
        self.entrada_datos_ic.pack(side="left", fill="both", expand=True)
        
        marco_boton_cargar = ttk.Frame(panel_izquierdo)
        marco_boton_cargar.pack(fill="x", pady=(0, 10))
        
        boton_cargar = ttk.Button(marco_boton_cargar, text="Cargar Desde Archivo", style="Modern.TButton",
                                  command=lambda: self.cargar_datos(destino=self.entrada_datos_ic))
        boton_cargar.pack(side="left")
        
        etiqueta_confianza = ttk.Label(panel_izquierdo, text="Nivel de confianza (%):", style="Modern.TLabel")
        etiqueta_confianza.pack(anchor="w", pady=(5, 2))
        
        self.entrada_confianza = ttk.Entry(panel_izquierdo, width=20, style="Modern.TEntry")
        self.entrada_confianza.insert(0, "95")
        self.entrada_confianza.pack(anchor="w", pady=(0, 10))
        
        etiqueta_tipo_prueba = ttk.Label(panel_izquierdo, text="Tipo de prueba:", style="Modern.TLabel")
        etiqueta_tipo_prueba.pack(anchor="w", pady=(5, 2))
        
        self.combo_tipo_prueba_ic = ttk.Combobox(panel_izquierdo, values=["Z (muestra grande)", "t (muestra pequeña)"], width=20)
        self.combo_tipo_prueba_ic.current(1)
        self.combo_tipo_prueba_ic.pack(anchor="w", pady=(0, 10))
        
        boton_calcular = ttk.Button(panel_izquierdo, text="Calcular Intervalo", style="Modern.TButton",
                             command=self.calcular_intervalo_confianza)
        boton_calcular.pack(pady=20)
        
        panel_derecho = ttk.LabelFrame(marco_principal, text="Resultados", padding=15)
        panel_derecho.pack(side="right", fill="both", expand=True, padx=(10, 0), pady=10)
        
        self.resultado_ic = scrolledtext.ScrolledText(panel_derecho, height=15, width=40)
        self.resultado_ic.pack(fill="both", expand=True)
        
        marco_boton_guardar = ttk.Frame(panel_derecho)
        marco_boton_guardar.pack(fill="x", pady=10)
        
        boton_guardar = ttk.Button(marco_boton_guardar, text="Guardar Resultados", style="Modern.TButton",
                                     command=lambda: self.guardar_resultados(fuente=self.resultado_ic))
        boton_guardar.pack(side="right")
        
    def crear_pestana_prueba_media(self):
        marco_principal = ttk.Frame(self.pestana2)
        marco_principal.pack(fill="both", expand=True, padx=20, pady=10)
        
        panel_izquierdo = ttk.LabelFrame(marco_principal, text="Datos de Entrada", padding=15)
        panel_izquierdo.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
        
        etiqueta_datos = ttk.Label(panel_izquierdo, text="Datos de la muestra (separados por comas):", style="Modern.TLabel")
        etiqueta_datos.pack(anchor="w", pady=(5, 2))
        
        self.marco_datos_prueba = ttk.Frame(panel_izquierdo)
        self.marco_datos_prueba.pack(fill="x", pady=(0, 10))
        
        self.entrada_datos_prueba = scrolledtext.ScrolledText(self.marco_datos_prueba, height=5, width=30)
        self.entrada_datos_prueba.pack(side="left", fill="both", expand=True)
        
        marco_boton_cargar = ttk.Frame(panel_izquierdo)
        marco_boton_cargar.pack(fill="x", pady=(0, 10))
        
        boton_cargar = ttk.Button(marco_boton_cargar, text="Cargar Desde Archivo", style="Modern.TButton",
                                  command=lambda: self.cargar_datos(destino=self.entrada_datos_prueba))
        boton_cargar.pack(side="left")
        
        etiqueta_nulo = ttk.Label(panel_izquierdo, text="Valor de la hipótesis nula:", style="Modern.TLabel")
        etiqueta_nulo.pack(anchor="w", pady=(5, 2))
        
        self.entrada_nulo = ttk.Entry(panel_izquierdo, width=20, style="Modern.TEntry")
        self.entrada_nulo.insert(0, "0")
        self.entrada_nulo.pack(anchor="w", pady=(0, 10))
        
        etiqueta_tipo_prueba = ttk.Label(panel_izquierdo, text="Tipo de prueba:", style="Modern.TLabel")
        etiqueta_tipo_prueba.pack(anchor="w", pady=(5, 2))
        
        self.combo_tipo_prueba = ttk.Combobox(panel_izquierdo, values=["Z (muestra grande)", "t (muestra pequeña)"], width=20)
        self.combo_tipo_prueba.current(1)
        self.combo_tipo_prueba.pack(anchor="w", pady=(0, 10))
        
        etiqueta_direccion = ttk.Label(panel_izquierdo, text="Hipótesis alternativa:", style="Modern.TLabel")
        etiqueta_direccion.pack(anchor="w", pady=(5, 2))
        
        self.combo_direccion = ttk.Combobox(panel_izquierdo, 
                                          values=["≠ (dos colas)", "< (cola izquierda)", "> (cola derecha)"], 
                                          width=20)
        self.combo_direccion.current(0)
        self.combo_direccion.pack(anchor="w", pady=(0, 10))
        
        etiqueta_alpha = ttk.Label(panel_izquierdo, text="Nivel de significancia:", style="Modern.TLabel")
        etiqueta_alpha.pack(anchor="w", pady=(5, 2))
        
        self.combo_alpha = ttk.Combobox(panel_izquierdo, values=["0.01", "0.05", "0.10"], width=20)
        self.combo_alpha.current(1)
        self.combo_alpha.pack(anchor="w", pady=(0, 10))
        
        boton_calcular = ttk.Button(panel_izquierdo, text="Realizar Prueba", style="Modern.TButton",
                             command=self.calcular_prueba_media)
        boton_calcular.pack(pady=10)
        
        panel_derecho = ttk.LabelFrame(marco_principal, text="Resultados de la Prueba", padding=15)
        panel_derecho.pack(side="right", fill="both", expand=True, padx=(10, 0), pady=10)
        
        self.resultado_prueba = scrolledtext.ScrolledText(panel_derecho, height=15, width=40)
        self.resultado_prueba.pack(fill="both", expand=True)
        
        marco_boton_guardar = ttk.Frame(panel_derecho)
        marco_boton_guardar.pack(fill="x", pady=10)
        
        boton_guardar = ttk.Button(marco_boton_guardar, text="Guardar Resultados", style="Modern.TButton",
                                     command=lambda: self.guardar_resultados(fuente=self.resultado_prueba))
        boton_guardar.pack(side="right")
        
    def mostrar_pestana(self, indice_pestana):
        self.cuaderno.select(indice_pestana)
        
    def calcular_intervalo_confianza(self):
        """Usa funciones modulares para cálculos."""
        try:
            datos_str = self.entrada_datos_ic.get("1.0", tk.END).strip()
            datos = validar_y_convertir_datos(datos_str)
            if datos is None:
                return

            confianza = float(self.entrada_confianza.get()) / 100
            tipo_prueba = self.combo_tipo_prueba_ic.get()

            if "Z" in tipo_prueba:
                resultado = calcular_intervalo_z(datos, confianza)
            else:
                resultado = calcular_intervalo_t(datos, confianza)

            estadisticas = resultado["estadisticas"]
            
            salida = f"""RESULTADOS DEL INTERVALO DE CONFIANZA (PRUEBA {estadisticas['prueba']})\n
Estadísticas descriptivas:
-------------------------
Tamaño de muestra (n): {estadisticas['n']}
Media muestral: {estadisticas['media']:.4f}
Desviación estándar: {estadisticas['desv_std']:.4f}
Error estándar: {estadisticas['error_std']:.4f}

Cálculo del intervalo:
---------------------
Nivel de confianza: {confianza*100:.1f}%
Valor crítico {estadisticas['prueba']}: {estadisticas['critico']:.4f}
Margen de error: {estadisticas['margen']:.4f}
{'Grados de libertad: ' + str(estadisticas['gl']) if estadisticas['prueba'] == 't' else ''}

Intervalo de confianza:
---------------------
IC {confianza*100:.1f}%: [{resultado['inferior']:.4f}, {resultado['superior']:.4f}]

Interpretación:
-------------
Con un {confianza*100:.1f}% de confianza, la media poblacional está entre {resultado['inferior']:.4f} y {resultado['superior']:.4f}."""

            self.resultado_ic.delete('1.0', tk.END)
            self.resultado_ic.insert(tk.END, salida)

        except Exception as e:
            messagebox.showerror("Error", f"Error en cálculo: {str(e)}")
            
    def calcular_prueba_media(self):
        """Usa funciones modulares para pruebas de hipótesis."""
        try:
            datos_str = self.entrada_datos_prueba.get('1.0', tk.END).strip()
            datos = validar_y_convertir_datos(datos_str)
            if datos is None:
                return

            valor_nulo = float(self.entrada_nulo.get())
            alpha = float(self.combo_alpha.get())
            direccion = self.combo_direccion.get()
            tipo_prueba = self.combo_tipo_prueba.get()

            if "Z" in tipo_prueba:
                resultado = realizar_prueba_z(datos, valor_nulo, alpha, direccion)
            else:
                resultado = realizar_prueba_t(datos, valor_nulo, alpha, direccion)

            estadisticas = resultado["estadisticas"]
            
            if resultado["valor_p"] <= alpha:
                decision = f"Se rechaza H₀: μ = {valor_nulo}"
                conclusion = f"Evidencia a favor de {estadisticas['hipotesis_alt']}"
            else:
                decision = f"No se rechaza H₀: μ = {valor_nulo}"
                conclusion = f"Sin evidencia para {estadisticas['hipotesis_alt']}"

            salida = f"""RESULTADOS DE LA PRUEBA DE HIPÓTESIS (PRUEBA {estadisticas['prueba'].upper()})\n
Estadísticas descriptivas:
-------------------------
Tamaño de muestra (n): {estadisticas['n']}
Media muestral: {estadisticas['media']:.4f}
Desviación estándar: {estadisticas['desv_std']:.4f}
Error estándar: {estadisticas['error_std']:.4f}

Planteamiento de hipótesis:
-------------------------
H₀: μ = {valor_nulo}
H₁: {estadisticas['hipotesis_alt']}

Cálculo de la prueba:
-------------------
Estadístico {estadisticas['prueba']}: {resultado['estadistico']:.4f}
Valor p: {resultado['valor_p']:.6f}
Nivel de significancia (α): {alpha}
{'Grados de libertad: ' + str(estadisticas['gl']) if estadisticas['prueba'] == 't' else ''}

Decisión estadística:
------------------
{decision}
{conclusion} (α = {alpha})"""

            self.resultado_prueba.delete('1.0', tk.END)
            self.resultado_prueba.insert(tk.END, salida)

        except Exception as e:
            messagebox.showerror("Error", f"Error en cálculo: {str(e)}")

    def cargar_datos(self, destino=None):
        ruta_archivo = filedialog.askopenfilename(
            title="Seleccionar archivo de datos",
            filetypes=[
                ("Archivos CSV", "*.csv"),
                ("Archivos Excel", "*.xlsx"),
                ("Archivos Parquet", "*.parquet"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if not ruta_archivo:
            return
            
        try:
            extension = os.path.splitext(ruta_archivo)[1].lower()
            
            if extension == '.csv':
                self.datos = pd.read_csv(ruta_archivo)
            elif extension == '.xlsx':
                self.datos = pd.read_excel(ruta_archivo)
            elif extension == '.parquet':
                self.datos = pd.read_parquet(ruta_archivo)
            else:
                messagebox.showerror("Error", "Formato de archivo no compatible")
                return
                
            if destino:
                destino.delete('1.0', tk.END)
                columnas_numericas = self.datos.select_dtypes(include=[np.number]).columns
                if len(columnas_numericas) > 0:
                    datos_numericos = self.datos[columnas_numericas[0]].dropna().tolist()
                    destino.insert(tk.END, ', '.join(map(str, datos_numericos)))
                else:
                    messagebox.showwarning("Advertencia", 
                                          "No se encontraron columnas numéricas en el archivo")
                    
            messagebox.showinfo("Éxito", f"Datos cargados correctamente: {len(self.datos)} registros")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar el archivo: {str(e)}")

    def guardar_resultados(self, fuente=None):
        if fuente:
            texto_a_guardar = fuente.get('1.0', tk.END)
        else:
            if self.cuaderno.index(self.cuaderno.select()) == 0:
                texto_a_guardar = self.resultado_ic.get('1.0', tk.END)
            else:
                texto_a_guardar = self.resultado_prueba.get('1.0', tk.END)
        
        if not texto_a_guardar.strip():
            messagebox.showwarning("Advertencia", "No hay resultados para guardar")
            return
            
        ruta_archivo = filedialog.asksaveasfilename(
            title="Guardar resultados",
            defaultextension=".txt",
            filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
        )
        
        if not ruta_archivo:
            return
            
        try:
            with open(ruta_archivo, 'w', encoding='utf-8') as archivo:
                archivo.write(texto_a_guardar)
            messagebox.showinfo("Éxito", "Resultados guardados correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar el archivo: {str(e)}")
            
    def mostrar_ayuda(self):
        ventana_ayuda = tk.Toplevel(self.raiz)
        ventana_ayuda.title("Ayuda - Conceptos Estadísticos")
        ventana_ayuda.geometry("650x500")
        ventana_ayuda.minsize(500, 400)
        
        marco = ttk.Frame(ventana_ayuda, padding=20)
        marco.pack(fill="both", expand=True)
        
        titulo = ttk.Label(marco, text="Conceptos Básicos de Estadística", 
                         font=("Arial", 14, "bold"))
        titulo.pack(pady=(0, 15))
        
        texto_ayuda = scrolledtext.ScrolledText(marco, wrap=tk.WORD, padx=10, pady=10,
                                            font=("Arial", 10))
        texto_ayuda.pack(fill="both", expand=True)
        
        contenido_ayuda = """
        # Intervalos de Confianza
        
        Un intervalo de confianza proporciona un rango de valores que probablemente contiene un parámetro poblacional desconocido.
        
        - **Nivel de confianza**: Representa el porcentaje de intervalos que contendrían el parámetro poblacional si el experimento se repitiera muchas veces.
        - **Fórmula general**: IC = Estimador ± (valor crítico × error estándar)
        
        ## Prueba Z vs Prueba t
        
        - **Prueba Z**: Se utiliza cuando se conoce la desviación estándar poblacional o cuando la muestra es grande (n ≥ 30).
        - **Prueba t**: Se utiliza cuando no se conoce la desviación estándar poblacional y el tamaño de la muestra es pequeño (n < 30).
        
        # Pruebas de Hipótesis
        
        Una prueba de hipótesis evalúa dos afirmaciones mutuamente excluyentes sobre una población.
        
        - **Hipótesis nula (H₀)**: Afirmación inicial que se asume verdadera (por ejemplo, μ = valor específico).
        - **Hipótesis alternativa (H₁)**: Afirmación contraria a la hipótesis nula.
        
        ## Tipos de Pruebas
        
        - **Prueba de dos colas (≠)**: Evalúa si el parámetro es diferente del valor especificado.
        - **Prueba de cola izquierda (<)**: Evalúa si el parámetro es menor que el valor especificado.
        - **Prueba de cola derecha (>)**: Evalúa si el parámetro es mayor que el valor especificado.
        
        ## Valor p
        
        El valor p representa la probabilidad de obtener un resultado al menos tan extremo como el observado, asumiendo que la hipótesis nula es verdadera.
        
        - Si valor p ≤ nivel de significancia (α): Se rechaza H₀
        - Si valor p > nivel de significancia (α): No se rechaza H₀
        
        # Interpretación de Resultados
        
        ## Para Intervalos de Confianza:
        
        Un intervalo de confianza al 95% significa que si tomáramos 100 muestras diferentes y construyéramos 100 intervalos de confianza, aproximadamente 95 de ellos contendrían el parámetro poblacional verdadero.
        
        ## Para Pruebas de Hipótesis:
        
        El rechazo de H₀ significa que hay evidencia estadística suficiente para apoyar la hipótesis alternativa.
        No rechazar H₀ significa que no hay evidencia estadística suficiente para rechazar la hipótesis nula.
        """
        
        texto_ayuda.insert(tk.END, contenido_ayuda)
        texto_ayuda.config(state=tk.DISABLED)
        
        boton_cerrar = ttk.Button(marco, text="Cerrar", command=ventana_ayuda.destroy)
        boton_cerrar.pack(pady=10)

if __name__ == "__main__":
    raiz = tk.Tk()
    app = AplicacionEstadistica(raiz)
    raiz.mainloop()