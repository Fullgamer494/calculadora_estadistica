import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import numpy as np
import pandas as pd
from scipy import stats
import os


def validar_y_convertir_datos(datos_str):
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
        messagebox.showerror("Error", f"Los Datos son inválidos: {str(e)}")
        return None

def validar_datos_muestra(datos, prueba_seleccionada):
    n = len(datos)
    if n < 2:
        raise ValueError("Se necesitan al menos 2 puntos de datos")
    
    if "Z" in prueba_seleccionada and n < 30:
        raise ValueError("La prueba Z requiere al menos 30 datos (use prueba t para muestras pequeñas)")
    
    return True

def calcular_intervalo_confianza_z(datos, confianza):
    "Intervalo de confianza Z con validación."
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
            "n": n, "media": media, "desv_std": desv_std, 
            "error_std": error_std, "critico": z_critico, 
            "margen": margen, "prueba": "Z"
        }
    }

def calcular_intervalo_confianza_t(datos, confianza):
    "Intervalo de confianza t con validación."
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
            "n": n, "media": media, "desv_std": desv_std, 
            "error_std": error_std, "critico": t_critico, 
            "margen": margen, "gl": n-1, "prueba": "t"
        }
    }

def realizar_prueba_hipotesis_z(datos, valor_nulo, alpha, direccion):
    "Prueba Z de hipótesis con validación."
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
    else:  # cola derecha
        valor_p = 1 - stats.norm.cdf(estadistico)
        hipotesis_alt = f"μ > {valor_nulo}"
    
    return {
        "estadistico": estadistico, 
        "valor_p": valor_p,
        "estadisticas": {
            "n": n, "media": media, "desv_std": desv_std, 
            "error_std": error_std, "hipotesis_alt": hipotesis_alt, 
            "prueba": "Z"
        }
    }

def realizar_prueba_hipotesis_t(datos, valor_nulo, alpha, direccion):
    "Prueba t de hipótesis con validación."
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
    else:  # cola derecha
        valor_p = 1 - stats.t.cdf(estadistico, df=gl)
        hipotesis_alt = f"μ > {valor_nulo}"
    
    return {
        "estadistico": estadistico, 
        "valor_p": valor_p,
        "estadisticas": {
            "n": n, "media": media, "desv_std": desv_std, 
            "error_std": error_std, "hipotesis_alt": hipotesis_alt, 
            "gl": gl, "prueba": "t"
        }
    }

def generar_resultados_intervalo(inferior, superior, estadisticas, confianza, widget_resultado):
    "Texto de resultados para intervalos de confianza."
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
IC {confianza*100:.1f}%: [{inferior:.4f}, {superior:.4f}]

Interpretación:
-------------
Con un {confianza*100:.1f}% de confianza, podemos afirmar que la media poblacional está en el rango de {inferior:.4f} y {superior:.4f}."""
    
    widget_resultado.delete('1.0', tk.END)
    widget_resultado.insert(tk.END, salida)

def generar_resultados_prueba(estadistico, valor_p, estadisticas, valor_nulo, alpha, widget_resultado):
    "Texto de resultados para pruebas de hipótesis."
    if valor_p <= alpha:
        decision = f"Se rechaza H₀: μ = {valor_nulo}"
        conclusion = f"Evidencia a favor de {estadisticas['hipotesis_alt']}"
    else:
        decision = f"No se rechaza H₀: μ = {valor_nulo}"
        conclusion = f"Sin evidencia para {estadisticas['hipotesis_alt']}"

    salida = f"""RESULTADOS DE LA PRUEBA DE HIPÓTESIS (PRUEBA {estadisticas['prueba'].upper()})\n
Estadísticas descriptivas:
----------------------------
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
Estadístico {estadisticas['prueba']}: {estadistico:.4f}
Valor p: {valor_p:.6f}
Nivel de significancia (α): {alpha}
{'Grados de libertad: ' + str(estadisticas['gl']) if estadisticas['prueba'] == 't' else ''}

Decisión estadística:
----------------------
{decision}
{conclusion} (α = {alpha})"""
    
    widget_resultado.delete('1.0', tk.END)
    widget_resultado.insert(tk.END, salida)


def configurar_estilos():
    "Estilos de la interfaz gráfica."
    colores = {
        "primario": "#000000",
        "secundario": "#000080",
        "acento": "#3498db",
        "hover": "#2980b9",
        "claro": "#ecf0f1",
        "texto": "#000000",
        "texto_claro": "#ecf0f1",
        "texto_negro": "#000000"
    }
    
    estilo = ttk.Style()
    
    estilo.configure("TNotebook", background=colores["claro"])
    estilo.configure("TNotebook.Tab", 
                    background=colores["claro"], 
                    foreground=colores["texto"], 
                    padding=[12, 8],
                    font=("Arial", 10, "bold"))
    estilo.map("TNotebook.Tab", 
              background=[("selected", colores["acento"])],
              foreground=[("selected", colores["texto_claro"])])
    
    estilo.configure("Panel.TFrame", background=colores["claro"], relief="raised")
    
    estilo.configure("Modern.TButton", background=colores["acento"], 
                     foreground=colores["texto_negro"], padding=10,
                     font=("Arial", 10, "bold"))
    estilo.map("Modern.TButton", 
               background=[("active", colores["hover"])],
               relief=[("pressed", "sunken")])
    
    estilo.configure("Modern.TEntry", padding=8, relief="flat")
    
    estilo.configure("Modern.TLabel", background=colores["claro"], 
                     foreground=colores["texto"],
                     font=("Arial", 10))
    
    estilo.configure("Title.TLabel", background=colores["claro"], 
                     foreground=colores["primario"],
                     font=("Arial", 12, "bold"))
    
    return colores

def crear_boton_menu(padre, opcion, colores):
    "Botón de menú personalizado."
    marco_boton = tk.Frame(padre, bg=colores["primario"])
    marco_boton.pack(fill="x", pady=2)
    
    def al_entrar(e):
        marco_boton.config(bg=colores["secundario"])
        etiqueta_texto.config(bg=colores["secundario"])
        
    def al_salir(e):
        marco_boton.config(bg=colores["primario"])
        etiqueta_texto.config(bg=colores["primario"])
        
    marco_boton.bind("<Enter>", al_entrar)
    marco_boton.bind("<Leave>", al_salir)
    marco_boton.bind("<Button-1>", lambda e: opcion["command"]())
    
    etiqueta_texto = tk.Label(marco_boton, text=opcion["text"], font=("Arial", 10),
                         fg=colores["texto_claro"], bg=colores["primario"],
                         anchor="w")
    etiqueta_texto.pack(side="left", fill="x", padx=5, pady=8)
    etiqueta_texto.bind("<Enter>", al_entrar)
    etiqueta_texto.bind("<Leave>", al_salir)
    etiqueta_texto.bind("<Button-1>", lambda e: opcion["command"]())
    
    return marco_boton

def crear_barra_lateral(contenedor_principal, colores, mostrar_pestana, guardar_resultados):
    "Side bar izquierda color negro acordarme"
    barra_lateral = tk.Frame(contenedor_principal, width=200, bg=colores["primario"])
    barra_lateral.pack(side="left", fill="y", padx=0, pady=0)
    barra_lateral.pack_propagate(False)
    
    marco_logo = tk.Frame(barra_lateral, bg=colores["primario"], height=120)
    marco_logo.pack(fill="x")
    
    etiqueta_titulo = tk.Label(marco_logo, text="ANÁLISIS\nESTADÍSTICO", font=("Arial", 14, "bold"),
                          fg=colores["texto_claro"], bg=colores["primario"])
    etiqueta_titulo.pack(fill="both", expand=True, pady=20)
    
    separador = ttk.Separator(barra_lateral, orient="horizontal")
    separador.pack(fill="x", padx=20)
    
    opciones_menu = [
        {"text": "Intervalos de Confianza", "command": lambda: mostrar_pestana(0)},
        {"text": "Pruebas de Media", "command": lambda: mostrar_pestana(1)},
        {"text": "Guardar Resultados", "command": guardar_resultados},
        {"text": "Ayuda", "command": lambda: mostrar_pestana(2)}
    ]
    
    contenedor_botones = tk.Frame(barra_lateral, bg=colores["primario"])
    contenedor_botones.pack(fill="both", expand=True, pady=10)
    
    for opcion in opciones_menu:
        crear_boton_menu(contenedor_botones, opcion, colores)
        
    pie_pagina = tk.Frame(barra_lateral, bg=colores["primario"], height=40)
    pie_pagina.pack(fill="x", side="bottom")
    
    etiqueta_version = tk.Label(pie_pagina, text="v1.0.2", font=("Arial", 8),
                            fg=colores["texto_claro"], bg=colores["primario"])
    etiqueta_version.pack(side="right", padx=10, pady=10)
    
    return barra_lateral

def crear_pestana_intervalo_confianza(pestana, colores):
    "Pestaña de intervalos de confianza."
    marco_principal = ttk.Frame(pestana)
    marco_principal.pack(fill="both", expand=True, padx=20, pady=10)
    
    panel_izquierdo = ttk.LabelFrame(marco_principal, text="Datos de Entrada", padding=15)
    panel_izquierdo.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
    
    etiqueta_datos = ttk.Label(panel_izquierdo, text="Datos de la muestra (separados por comas):", style="Modern.TLabel")
    etiqueta_datos.pack(anchor="w", pady=(5, 2))
    
    marco_datos_ic = ttk.Frame(panel_izquierdo)
    marco_datos_ic.pack(fill="x", pady=(0, 10))
    
    entrada_datos_ic = scrolledtext.ScrolledText(marco_datos_ic, height=5, width=30)
    entrada_datos_ic.pack(side="left", fill="both", expand=True)
    
    marco_boton_cargar = ttk.Frame(panel_izquierdo)
    marco_boton_cargar.pack(fill="x", pady=(0, 10))
    
    boton_cargar = ttk.Button(marco_boton_cargar, text="Cargar Desde Archivo", style="Modern.TButton")
    boton_cargar.pack(side="left")
    
    etiqueta_confianza = ttk.Label(panel_izquierdo, text="Nivel de confianza (%):", style="Modern.TLabel")
    etiqueta_confianza.pack(anchor="w", pady=(5, 2))
    
    entrada_confianza = ttk.Entry(panel_izquierdo, width=20, style="Modern.TEntry")
    entrada_confianza.insert(0, "95")
    entrada_confianza.pack(anchor="w", pady=(0, 10))
    
    etiqueta_tipo_prueba = ttk.Label(panel_izquierdo, text="Tipo de prueba:", style="Modern.TLabel")
    etiqueta_tipo_prueba.pack(anchor="w", pady=(5, 2))
    
    combo_tipo_prueba_ic = ttk.Combobox(panel_izquierdo, values=["Z (muestra grande)", "t (muestra pequeña)"], width=20)
    combo_tipo_prueba_ic.current(1)
    combo_tipo_prueba_ic.pack(anchor="w", pady=(0, 10))
    
    boton_calcular = ttk.Button(panel_izquierdo, text="Calcular Intervalo", style="Modern.TButton")
    boton_calcular.pack(pady=20)
    
    panel_derecho = ttk.LabelFrame(marco_principal, text="Resultados", padding=15)
    panel_derecho.pack(side="right", fill="both", expand=True, padx=(10, 0), pady=10)
    
    resultado_ic = scrolledtext.ScrolledText(panel_derecho, height=15, width=40)
    resultado_ic.pack(fill="both", expand=True)
    
    marco_boton_guardar = ttk.Frame(panel_derecho)
    marco_boton_guardar.pack(fill="x", pady=10)
    
    boton_guardar = ttk.Button(marco_boton_guardar, text="Guardar Resultados", style="Modern.TButton")
    boton_guardar.pack(side="right")
    
    return {
        "entrada_datos": entrada_datos_ic,
        "entrada_confianza": entrada_confianza,
        "combo_tipo_prueba": combo_tipo_prueba_ic,
        "resultado": resultado_ic,
        "boton_calcular": boton_calcular,
        "boton_cargar": boton_cargar,
        "boton_guardar": boton_guardar
    }

def crear_pestana_prueba_media(pestana, colores):
    "Pestaña de pruebas de media."
    marco_principal = ttk.Frame(pestana)
    marco_principal.pack(fill="both", expand=True, padx=20, pady=10)
    
    panel_izquierdo = ttk.LabelFrame(marco_principal, text="Datos de Entrada", padding=15)
    panel_izquierdo.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
    
    etiqueta_datos = ttk.Label(panel_izquierdo, text="Datos de la muestra (separados por comas):", style="Modern.TLabel")
    etiqueta_datos.pack(anchor="w", pady=(5, 2))
    
    marco_datos_prueba = ttk.Frame(panel_izquierdo)
    marco_datos_prueba.pack(fill="x", pady=(0, 10))
    
    entrada_datos_prueba = scrolledtext.ScrolledText(marco_datos_prueba, height=5, width=30)
    entrada_datos_prueba.pack(side="left", fill="both", expand=True)
    
    marco_boton_cargar = ttk.Frame(panel_izquierdo)
    marco_boton_cargar.pack(fill="x", pady=(0, 10))
    
    boton_cargar = ttk.Button(marco_boton_cargar, text="Cargar Desde Archivo", style="Modern.TButton")
    boton_cargar.pack(side="left")
    
    etiqueta_nulo = ttk.Label(panel_izquierdo, text="Valor de la hipótesis nula:", style="Modern.TLabel")
    etiqueta_nulo.pack(anchor="w", pady=(5, 2))
    
    entrada_nulo = ttk.Entry(panel_izquierdo, width=20, style="Modern.TEntry")
    entrada_nulo.insert(0, "0")
    entrada_nulo.pack(anchor="w", pady=(0, 10))
    
    etiqueta_tipo_prueba = ttk.Label(panel_izquierdo, text="Tipo de prueba:", style="Modern.TLabel")
    etiqueta_tipo_prueba.pack(anchor="w", pady=(5, 2))
    
    combo_tipo_prueba = ttk.Combobox(panel_izquierdo, values=["Z (muestra grande)", "t (muestra pequeña)"], width=20)
    combo_tipo_prueba.current(1)
    combo_tipo_prueba.pack(anchor="w", pady=(0, 10))
    
    etiqueta_direccion = ttk.Label(panel_izquierdo, text="Hipótesis alternativa:", style="Modern.TLabel")
    etiqueta_direccion.pack(anchor="w", pady=(5, 2))
    
    combo_direccion = ttk.Combobox(panel_izquierdo, 
                                  values=["≠ (dos colas)", "< (cola izquierda)", "> (cola derecha)"], 
                                  width=20)
    combo_direccion.current(0)
    combo_direccion.pack(anchor="w", pady=(0, 10))
    
    etiqueta_alpha = ttk.Label(panel_izquierdo, text="Nivel de significancia:", style="Modern.TLabel")
    etiqueta_alpha.pack(anchor="w", pady=(5, 2))
    
    combo_alpha = ttk.Combobox(panel_izquierdo, values=["0.01", "0.05", "0.10"], width=20)
    combo_alpha.current(1)
    combo_alpha.pack(anchor="w", pady=(0, 10))
    
    boton_calcular = ttk.Button(panel_izquierdo, text="Realizar Prueba", style="Modern.TButton")
    boton_calcular.pack(pady=10)
    
    panel_derecho = ttk.LabelFrame(marco_principal, text="Resultados de la Prueba", padding=15)
    panel_derecho.pack(side="right", fill="both", expand=True, padx=(10, 0), pady=10)
    
    resultado_prueba = scrolledtext.ScrolledText(panel_derecho, height=15, width=40)
    resultado_prueba.pack(fill="both", expand=True)
    
    marco_boton_guardar = ttk.Frame(panel_derecho)
    marco_boton_guardar.pack(fill="x", pady=10)
    
    boton_guardar = ttk.Button(marco_boton_guardar, text="Guardar Resultados", style="Modern.TButton")
    boton_guardar.pack(side="right")
    
    return {
        "entrada_datos": entrada_datos_prueba,
        "entrada_nulo": entrada_nulo,
        "combo_tipo_prueba": combo_tipo_prueba,
        "combo_direccion": combo_direccion,
        "combo_alpha": combo_alpha,
        "resultado": resultado_prueba,
        "boton_calcular": boton_calcular,
        "boton_cargar": boton_cargar,
        "boton_guardar": boton_guardar
    }

def crear_pestana_ayuda(pestana):
    "Pestaña de ayuda."
    marco_principal = ttk.Frame(pestana)
    marco_principal.pack(fill="both", expand=True, padx=20, pady=10)
    
    titulo = ttk.Label(marco_principal, text="Conceptos Básicos de Estadística", 
                     font=("Arial", 14, "bold"))
    titulo.pack(pady=(0, 15))
    
    texto_ayuda = scrolledtext.ScrolledText(marco_principal, wrap=tk.WORD, padx=10, pady=10,
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
    
    Un intervalo de confianza al 95% significa que si tomáramos 100 muestras diferentes y construyéramos 100 intervalos de confianza, aproximadamente 95 de ellos contendrían el parámetro poblacional.
    
    ## Para Pruebas de Hipótesis:
    
    El rechazo de H₀ significa que hay evidencia estadística suficiente para apoyar la hipótesis alternativa.
    No rechazar H₀ significa que no hay evidencia estadística suficiente para rechazar la hipótesis nula.
    """
    
    texto_ayuda.insert(tk.END, contenido_ayuda)
    texto_ayuda.config(state=tk.DISABLED)

def crear_pestanas(contenido, colores):
    "Pestañas principales de la aplicación."
    cuaderno = ttk.Notebook(contenido)
    cuaderno.pack(fill="both", expand=True, padx=5, pady=5)
    
    pestana1 = ttk.Frame(cuaderno, style="Panel.TFrame")
    cuaderno.add(pestana1, text="Intervalos de Confianza")
    widgets_ic = crear_pestana_intervalo_confianza(pestana1, colores)
    
    pestana2 = ttk.Frame(cuaderno, style="Panel.TFrame")
    cuaderno.add(pestana2, text="Pruebas de Media")
    widgets_prueba = crear_pestana_prueba_media(pestana2, colores)
    
    pestana3 = ttk.Frame(cuaderno, style="Panel.TFrame")
    cuaderno.add(pestana3, text="Ayuda")
    crear_pestana_ayuda(pestana3)
    
    return cuaderno, widgets_ic, widgets_prueba

def cargar_datos(destino=None):
    "Carga datos desde un archivo."
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
            datos = pd.read_csv(ruta_archivo)
        elif extension == '.xlsx':
            datos = pd.read_excel(ruta_archivo)
        elif extension == '.parquet':
            datos = pd.read_parquet(ruta_archivo)
        else:
            messagebox.showerror("Error", "Formato de archivo no compatible")
            return
            
        if destino:
            destino.delete('1.0', tk.END)
            columnas_numericas = datos.select_dtypes(include=[np.number]).columns
            if len(columnas_numericas) > 0:
                datos_numericos = datos[columnas_numericas[0]].dropna().tolist()
                destino.insert(tk.END, ', '.join(map(str, datos_numericos)))
            else:
                messagebox.showwarning("Advertencia", 
                                      "No se encontraron columnas numéricas en el archivo")
                
        messagebox.showinfo("Éxito", f"Datos cargados correctamente: {len(datos)} registros")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error al cargar el archivo: {str(e)}")

def guardar_resultados(fuente=None):
    "Resultados en un archivo."
    if fuente:
        texto_a_guardar = fuente.get('1.0', tk.END)
    else:
        messagebox.showwarning("Advertencia", "No hay resultados para guardar")
        return
    
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

def main():
    "Función principal de aplicación."
    raiz = tk.Tk()
    raiz.title("Análisis Estadístico")
    raiz.geometry("900x600")
    raiz.minsize(700, 500)
    
    # Configurar estilos y colores
    colores = configurar_estilos()
    
    # Crear contenedor principal
    contenedor_principal = ttk.Frame(raiz)
    contenedor_principal.pack(fill="both", expand=True)
    
    # Funciones con acceso a widgets necesarios
    def mostrar_pestana(indice_pestana):
        cuaderno.select(indice_pestana)
    
    # Crear barra lateral
    crear_barra_lateral(contenedor_principal, colores, mostrar_pestana, guardar_resultados)
    
    # Crear contenido principal
    contenido = ttk.Frame(contenedor_principal, style="Panel.TFrame")
    contenido.pack(side="right", fill="both", expand=True, padx=10, pady=10)
    
    # Crear pestañas y widgets
    cuaderno, widgets_ic, widgets_prueba = crear_pestanas(contenido, colores)
    
    # Configurar comandos de los botones
    def calcular_ic():
        try:
            datos_str = widgets_ic["entrada_datos"].get("1.0", tk.END).strip()
            datos = validar_y_convertir_datos(datos_str)
            if datos is None:
                return

            confianza = float(widgets_ic["entrada_confianza"].get()) / 100
            tipo_prueba = widgets_ic["combo_tipo_prueba"].get()
            
            validar_datos_muestra(datos, tipo_prueba)

            if "Z" in tipo_prueba:
                resultado = calcular_intervalo_confianza_z(datos, confianza)
            else:
                resultado = calcular_intervalo_confianza_t(datos, confianza)

            generar_resultados_intervalo(
                resultado["inferior"], 
                resultado["superior"], 
                resultado["estadisticas"], 
                confianza, 
                widgets_ic["resultado"]
            )

        except ValueError as e:
            messagebox.showerror("Error de validación", str(e))
        except Exception as e:
            messagebox.showerror("Error en cálculo", f"Error técnico: {str(e)}")

    def calcular_prueba():
        try:
            datos_str = widgets_prueba["entrada_datos"].get('1.0', tk.END).strip()
            datos = validar_y_convertir_datos(datos_str)
            if datos is None:
                return

            valor_nulo = float(widgets_prueba["entrada_nulo"].get())
            alpha = float(widgets_prueba["combo_alpha"].get())
            direccion = widgets_prueba["combo_direccion"].get()
            tipo_prueba = widgets_prueba["combo_tipo_prueba"].get()
            
            validar_datos_muestra(datos, tipo_prueba)

            if "Z" in tipo_prueba:
                resultado = realizar_prueba_hipotesis_z(datos, valor_nulo, alpha, direccion)
            else:
                resultado = realizar_prueba_hipotesis_t(datos, valor_nulo, alpha, direccion)

            generar_resultados_prueba(
                resultado["estadistico"], 
                resultado["valor_p"], 
                resultado["estadisticas"], 
                valor_nulo, 
                alpha, 
                widgets_prueba["resultado"]
            )

        except ValueError as e:
            messagebox.showerror("Error de validación", str(e))
        except Exception as e:
            messagebox.showerror("Error en cálculo", f"Error técnico: {str(e)}")

    # Configurar eventos
    widgets_ic["boton_calcular"].config(command=calcular_ic)
    widgets_ic["boton_cargar"].config(command=lambda: cargar_datos(widgets_ic["entrada_datos"]))
    widgets_ic["boton_guardar"].config(command=lambda: guardar_resultados(widgets_ic["resultado"]))
    
    widgets_prueba["boton_calcular"].config(command=calcular_prueba)
    widgets_prueba["boton_cargar"].config(command=lambda: cargar_datos(widgets_prueba["entrada_datos"]))
    widgets_prueba["boton_guardar"].config(command=lambda: guardar_resultados(widgets_prueba["resultado"]))
    
    raiz.mainloop()

if __name__ == "__main__":
    main()