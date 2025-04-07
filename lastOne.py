import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import numpy as np
import scipy.stats as stats
import pandas as pd
import os
import tkinter.font as tkfont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import norm, t
from tkhtmlview import HTMLLabel

def parse_data(data_str):
    """Convierte una cadena de datos separados por comas a una lista de números"""
    try:
        # Reemplazar varios separadores posibles y eliminar espacios
        clean_data = data_str.replace(';', ',').replace('\t', ',').replace('\n', ',')
        data_list = [float(x.strip()) for x in clean_data.split(',') if x.strip()]
        return np.array(data_list)
    except ValueError:
        messagebox.showerror("Error", "Los datos ingresados no son válidos. Deben ser números separados por comas.")
        return None

def plot_distribution(test_stat, critical_value, test_type, direction, n, frame):
    """Grafica la distribución t-student o normal Z con los valores críticos y el valor de prueba y la integra en un frame de tkinter"""
    x_values = np.linspace(-4, 4, 1000)
    
    fig, ax = plt.subplots()
    
    # Configurar el color de fondo y el contraste de las letras
    fig.patch.set_facecolor('#403d39')
    ax.set_facecolor('#403d39')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    
    if "Z" in test_type:
        # Distribución Z (normal estándar)
        y_values = norm.pdf(x_values)
        ax.plot(x_values, y_values, label='Distribución Z', color='#f08c00', linewidth=4)
    else:
        # Distribución t
        df = n - 1
        y_values = t.pdf(x_values, df)
        ax.plot(x_values, y_values, label=f'Distribución t (df={df})', color='#f08c00', linewidth=4)
        
    # Graficar el valor crítico y el valor de prueba
    ax.axvline(x=test_stat, color='#197278', linestyle='--', label=f'Estadístico de prueba ({test_stat:.2f})', linewidth=2)
    
    if direction == "Dos colas":
        ax.axvline(x=critical_value, color='#9e2a2b', linestyle='--', linewidth=2, label=f'Valor crítico ({critical_value:.2f})')
        ax.axvline(x=-critical_value, color='#9e2a2b', linestyle='--', linewidth=2)
        ax.fill_between(x_values, 0, y_values, where=(x_values <= -critical_value) | (x_values >= critical_value), color='#9e2a2b', alpha=0.3)
    elif direction == "Cola izquierda":
        ax.axvline(x=critical_value, color='#9e2a2b', linestyle='--', linewidth=2, label=f'Valor crítico ({critical_value:.2f})')
        ax.fill_between(x_values, 0, y_values, where=(x_values <= critical_value), color='#9e2a2b', alpha=0.3)
    else:  # Cola derecha
        ax.axvline(x=critical_value, color='#9e2a2b', linestyle='--', linewidth=2, label=f'Valor crítico ({critical_value:.2f})')
        ax.fill_between(x_values, 0, y_values, where=(x_values >= critical_value), color='#9e2a2b', alpha=0.3)
    
    ax.legend(facecolor='#000000', edgecolor='white', framealpha=0.5)
    for text in ax.legend().get_texts():
        text.set_color('#000000')
        
    ax.set_title('Distribución de prueba con valores críticos y estadístico de prueba', color='white', fontsize=12)
    ax.set_xlabel('Valor de la variable', color='white', fontsize=11)
    ax.set_ylabel('Densidad de probabilidad', color='white', fontsize=11)

    # Crear el canvas de matplotlib y agregarlo al frame de tkinter
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def calculate_confidence_interval(data_entry, conf_level_entry, test_type_combobox, results_text, graph_frame):
    """Calcula el intervalo de confianza para la media"""
    data_str = data_entry.get()
    
    if not data_str:
        messagebox.showerror("Error", "Ingrese los datos de la muestra")
        return
        
    data = parse_data(data_str)
    if data is None:
        return
        
    try:
        conf_level = float(conf_level_entry.get())
        if conf_level <= 0 or conf_level >= 100:
            messagebox.showerror("Error", "El nivel de confianza debe estar entre 0 y 100")
            return
            
        alpha = (100 - conf_level) / 100
        n = len(data)
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)  # ddof=1 para usar la desviación estándar muestral
        
        test_type = test_type_combobox.get()
        
        if "Z" in test_type:
            # Prueba Z (asumiendo que std_dev es la desviación estándar poblacional)
            std_error = std_dev / np.sqrt(n)
            critical_value = stats.norm.ppf(1 - alpha/2)
            margin_error = critical_value * std_error
            lower_bound = mean - margin_error
            upper_bound = mean + margin_error
            distribution = "normal estándar (Z)"
        else:
            # Prueba t
            std_error = std_dev / np.sqrt(n)
            critical_value = stats.t.ppf(1 - alpha/2, n-1)
            margin_error = critical_value * std_error
            lower_bound = mean - margin_error
            upper_bound = mean + margin_error
            distribution = f"t-Student con {n-1} grados de libertad"
            
        # Mostrar resultados
        results_text_content = f"""
📊 Intervalo de Confianza para la Media
    📥 Datos de entrada

        🧮 Tamaño de muestra (n): {n}

        📈 Media muestral (x̄): {mean:.6f}

        📉 Desviación estándar muestral (s): {std_dev:.6f}

        🎯 Nivel de confianza: {conf_level:.1f}%

        🧪 Tipo de prueba: {test_type}

    🔍 Cálculos

        🧾 Error estándar: {std_error:.6f}

        📏 Valor crítico ({distribution}): {critical_value:.6f}

        📐 Margen de error: {margin_error:.6f}

    ✅ Resultado

        📌 Intervalo de confianza al {conf_level:.1f}%: [{lower_bound:.6f}, {upper_bound:.6f}]

        💡 Interpretación: Con un nivel de confianza del {conf_level:.1f}%, se estima que la media poblacional se encuentra entre {lower_bound:.6f} y {upper_bound:.6f}.
        """
        
        results_text.delete(1.0, tk.END)
        results_text.insert(tk.INSERT, results_text_content)
        
        # Limpiar el frame de la gráfica antes de dibujar
        for widget in graph_frame.winfo_children():
            widget.destroy()
        
        # Llamar a la función de graficado
        plot_distribution(0, critical_value, test_type, "Dos colas", n, graph_frame)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error en los cálculos: {str(e)}")

def calculate_hypothesis_test(data_entry, null_hypo_entry, alpha_entry, test_type_combobox, 
                              direction_combobox, results_text, graph_frame):
    """Realiza una prueba de hipótesis para la media"""
    data_str = data_entry.get()
    
    if not data_str:
        messagebox.showerror("Error", "Ingrese los datos de la muestra")
        return
        
    data = parse_data(data_str)
    if data is None:
        return
        
    try:
        null_value = float(null_hypo_entry.get())
        alpha = float(alpha_entry.get())
        
        if alpha <= 0 or alpha >= 1:
            messagebox.showerror("Error", "El nivel de significancia (α) debe estar entre 0 y 1")
            return
            
        n = len(data)
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)
        
        test_type = test_type_combobox.get()
        direction = direction_combobox.get()
        
        # Calcular el estadístico de prueba
        std_error = std_dev / np.sqrt(n)
        test_stat = (mean - null_value) / std_error
        
        # Calcular el valor p según el tipo de prueba y dirección
        if "Z" in test_type:
            # Prueba Z
            if direction == "Dos colas":
                p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
                critical_value = stats.norm.ppf(1 - alpha/2)
                hypothesis_alt = f"μ ≠ {null_value}"
            elif direction == "Cola izquierda":
                p_value = stats.norm.cdf(test_stat)
                critical_value = stats.norm.ppf(alpha)
                hypothesis_alt = f"μ < {null_value}"
            else:  # Cola derecha
                p_value = 1 - stats.norm.cdf(test_stat)
                critical_value = stats.norm.ppf(1 - alpha)
                hypothesis_alt = f"μ > {null_value}"
                
            distribution = "distribución normal estándar (Z)"
            
        else:
            # Prueba t
            df = n - 1
            if direction == "Dos colas":
                p_value = 2 * (1 - stats.t.cdf(abs(test_stat), df))
                critical_value = stats.t.ppf(1 - alpha/2, df)
                hypothesis_alt = f"μ ≠ {null_value}"
            elif direction == "Cola izquierda":
                p_value = stats.t.cdf(test_stat, df)
                critical_value = stats.t.ppf(alpha, df)
                hypothesis_alt = f"μ < {null_value}"
            else:  # Cola derecha
                p_value = 1 - stats.t.cdf(test_stat, df)
                critical_value = stats.t.ppf(1 - alpha, df)
                hypothesis_alt = f"μ > {null_value}"
                
            distribution = f"distribución t con {df} grados de libertad"
            
        # Decisión de la prueba
        if direction == "Dos colas":
            reject = abs(test_stat) > abs(critical_value)
        elif direction == "Cola izquierda":
            reject = test_stat < critical_value
        else:  # Cola derecha
            reject = test_stat > critical_value
            
        decision = "Se rechaza" if reject else "No se rechaza"
        
        # Mostrar resultados
        result_text = f"""
🧪 Prueba de Hipótesis para la Media
    📥 Datos de Entrada

        🔢 Tamaño de muestra (n): {n}

        📊 Media muestral (x̄): {mean:.6f}

        📈 Desviación estándar muestral (s): {std_dev:.6f}

        🎯 Valor de la hipótesis nula (μ₀): {null_value}

        ⚠️ Nivel de significancia (α): {alpha}

        🧭 Tipo de prueba: {test_type}

        ↔️ Dirección: {direction}

        🧾 Hipótesis

            H₀: μ {"=" if direction == "Dos colas" else "≥" if direction == "Cola izquierda" else "≤"} {null_value}

            H₁: {hypothesis_alt}

    🧮 Cálculos

        🧠 Error estándar (SE): {std_error:.6f}

        📏 Estadístico de prueba: {test_stat:.6f}

        🎯 Valor crítico ({distribution}): {critical_value:.6f}

        📉 Valor p: {p_value:.6f}

    ✅ Resultado

        📝 Decisión: {decision} la hipótesis nula al nivel de significancia α = {alpha}

        📌 Interpretación: Se {decision} la hipótesis de que la media poblacional {"es igual a" if direction == "Dos colas" else "es mayor o igual a" if direction == "Cola izquierda" else "es menor o igual a"} {null_value}.
        """        
        results_text.delete(1.0, tk.END)
        results_text.insert(tk.INSERT, result_text)

        # Limpiar el frame de la gráfica antes de dibujar
        for widget in graph_frame.winfo_children():
            widget.destroy()
        
        # Llamar a la función de graficado
        plot_distribution(test_stat, critical_value, test_type, direction, n, graph_frame)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error en los cálculos: {str(e)}")

def load_data(ventana, conf_data_entry=None, hypo_data_entry=None):
    """Carga datos desde un archivo"""
    filetypes = [
        ("Archivos CSV", "*.csv"),
        ("Archivos Excel", "*.xlsx"),
        ("Archivos Parquet", "*.parquet"),
        ("Todos los archivos", "*.*")
    ]
    
    filename = filedialog.askopenfilename(title="Seleccionar archivo de datos", filetypes=filetypes)
    
    if not filename:
        return
        
    try:
        # Leer el archivo según su extensión
        extension = os.path.splitext(filename)[1].lower()
        
        if extension == '.csv':
            data = pd.read_csv(filename)
        elif extension == '.xlsx':
            data = pd.read_excel(filename)
        elif extension == '.parquet':
            data = pd.read_parquet(filename)
        else:
            messagebox.showerror("Error", "Formato de archivo no soportado")
            return
            
        # Verificar que haya datos numéricos
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            messagebox.showerror("Error", "No se encontraron columnas numéricas en el archivo")
            return
            
        # Si hay más de una columna numérica, preguntar cuál usar
        selected_col = None
        if len(numeric_cols) > 1:
            col_select_window = tk.Toplevel(ventana)
            col_select_window.title("Seleccionar columna")
            col_select_window.geometry("300x200")
            
            # Variable para almacenar la columna seleccionada
            selected_var = tk.StringVar(value=numeric_cols[0])
            
            tk.Label(col_select_window, text="Seleccione la columna con los datos:").grid(row=0, column=0, padx=10, pady=10)
            
            for i, col_name in enumerate(numeric_cols):
                tk.Radiobutton(col_select_window, text=col_name, variable=selected_var, value=col_name).grid(row=i+1, column=0, sticky="w", padx=20)
                
            def confirm_selection():
                nonlocal selected_col
                selected_col = selected_var.get()
                col_select_window.destroy()
                
            tk.Button(col_select_window, text="Seleccionar", command=confirm_selection).grid(row=len(numeric_cols)+1, column=0, pady=10)
            
            ventana.wait_window(col_select_window)
            
            if not selected_col:  # Si no se seleccionó nada
                return
                
            selected_data = data[selected_col].dropna().tolist()
        else:
            selected_data = data[numeric_cols[0]].dropna().tolist()
            
        # Convertir la lista a una cadena separada por comas
        data_str = ", ".join(map(str, selected_data))
        
        # Actualizar el campo correspondiente según la pestaña
        if conf_data_entry is not None:
            conf_data_entry.delete(0, tk.END)
            conf_data_entry.insert(0, data_str)
            
        if hypo_data_entry is not None:
            hypo_data_entry.delete(0, tk.END)
            hypo_data_entry.insert(0, data_str)
            
        messagebox.showinfo("Éxito", f"Se cargaron {len(selected_data)} datos con éxito")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error al cargar el archivo: {str(e)}")

def save_results(results_text, title=""):
    """Guarda los resultados en un archivo de texto"""
    filetypes = [("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
    filename = filedialog.asksaveasfilename(title="Guardar resultados", defaultextension=".txt", filetypes=filetypes)
    
    if not filename:
        return
        
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"{title}\n\n")
            file.write(results_text.get(1.0, tk.END))
                
        messagebox.showinfo("Éxito", f"Resultados guardados en {filename}")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error al guardar los resultados: {str(e)}")

def setup_confidence_interval_tab(tab):
    """Configura la pestaña de intervalos de confianza"""
    # Variables para almacenar los widgets que necesitarán ser accedidos
    conf_widgets = {}
    
    # Crear un estilo personalizado para las etiquetas
    estilo_labels = ttk.Style()
    estilo_labels.configure("stLabs.TLabel", 
                        font=("Arial", 12),
                        foreground="#333333",
                        background="#ccc5b9",
                        padding=(10, 5))

    # Crear un estilo personalizado para los combobox
    estilos_combo = ttk.Style()
    estilos_combo.configure("stCombos.TCombobox",
                        foreground="#333333",
                        background="#ccc5b9",
                        padding=(10, 5))

    # Crear un estilo personalizado para los entry
    estilo_entry = {
        "font": ("Arial", 12),
        "fg": "#333333",  # color del texto
        "insertbackground": "#333333",  # color del cursor
        "relief": "flat",  # tipo de borde (solid, raised, sunken, ridge, groove, flat)
        "borderwidth": 10,
        "highlightthickness": 1,
        "highlightbackground": "#FFFFFF",  # Color del borde igual al fondo
        "highlightcolor": "#ccc5b9"  # Color del resaltado igual al fondo
    }

    # Crear un estilo para el botón
    estilo_boton = ttk.Style()
    estilo_boton.configure("stBttn.TButton",
                foreground="#FFFFFF",
                font=("Arial", 12),
                justify="center",
                padding=(10, 5),
                relief= "raised",
                background="#197278")
    
    # Configurar el grid para que se expanda
    tab.grid_rowconfigure(0, weight=1, minsize=300)
    tab.grid_rowconfigure(1, weight=1, minsize=350)
    
    tab.grid_columnconfigure(0, weight=1)
    
    # Contenedor para los widgets de entrada y salida de datos
    conf_widgets['frame'] = tk.Frame(tab, bg="#ccc5b9")
    conf_widgets['frame'].grid(row=0, column=0, sticky="new", padx=10, pady=4)
    
    conf_widgets['frame'].grid_rowconfigure(0, weight=3)
    conf_widgets['frame'].grid_rowconfigure(1, weight=1)
    conf_widgets['frame'].grid_rowconfigure(2, weight=1)
    conf_widgets['frame'].grid_rowconfigure(3, weight=1)
    conf_widgets['frame'].grid_rowconfigure(4, weight=1)
    conf_widgets['frame'].grid_rowconfigure(5, weight=1)
    conf_widgets['frame'].grid_rowconfigure(6, weight=3)
    
    conf_widgets['frame'].grid_columnconfigure(0, weight=1)
    conf_widgets['frame'].grid_columnconfigure(1, weight=1)
    conf_widgets['frame'].grid_columnconfigure(2, weight=1)
    conf_widgets['frame'].grid_columnconfigure(3, weight=5)
    
    # Etiqueta y campo para los datos de muestra
    ttk.Label(conf_widgets['frame'], text="Datos de la muestra (separados por comas):", style="stLabs.TLabel").grid(row=1, column=0, sticky="w", padx=10)
    conf_widgets['data_entry'] = tk.Entry(conf_widgets['frame'], **estilo_entry)
    conf_widgets['data_entry'].grid(row=1, column=1, sticky="ew")
    
    # Nivel de confianza
    ttk.Label(conf_widgets['frame'], text="Nivel de confianza (%):", style="stLabs.TLabel").grid(row=2, column=0, sticky="w", padx=10)
    conf_widgets['conf_level_entry'] = tk.Entry(conf_widgets['frame'], **estilo_entry)
    conf_widgets['conf_level_entry'].insert(0, "95")
    conf_widgets['conf_level_entry'].grid(row=2, column=1, sticky="ew")
    
    custom_font = tkfont.Font(family="Arial", size=12)
    # Tipo de prueba
    ttk.Label(conf_widgets['frame'], text="Tipo de prueba:", style="stLabs.TLabel").grid(row=3, column=0, sticky="w", padx=10)
    conf_widgets['test_type'] = ttk.Combobox(conf_widgets['frame'], values=["Z (muestra grande o varianza conocida)", "t (muestra pequeña)"], style="stCombos.TCombobox", font=custom_font)
    conf_widgets['test_type'].current(1)  # Seleccionar prueba t por defecto
    conf_widgets['test_type'].grid(row=3, column=1, sticky="ew")
    
    # Area de resultados
    fuente_personalizada = tkfont.Font(family="Arial", size=12)
    ttk.Label(conf_widgets['frame'], text="Resultados:", style="stLabs.TLabel").grid(row=1, column=3, sticky="nsew", padx=10)
    conf_widgets['results'] = scrolledtext.ScrolledText(conf_widgets['frame'], font=fuente_personalizada)
    conf_widgets['results'].grid(row=2, rowspan=3, column=3, padx=5)

    # Botones
    load_button = tk.Button(conf_widgets['frame'], text="  ⬆  ", font=('Arial', 12), command=lambda: load_data(conf_widgets['frame'], conf_widgets['data_entry'], None))
    load_button.config(background="#197278", foreground="#FFFFFF", relief="raised")
    load_button.grid(row=1, column=2, sticky="nsew")

    calc_button = ttk.Button(conf_widgets['frame'], text="Calcular", style="stBttn.TButton",
                           command=lambda: calculate_confidence_interval(
                               conf_widgets['data_entry'], 
                               conf_widgets['conf_level_entry'], 
                               conf_widgets['test_type'], 
                               conf_widgets['results'], 
                               conf_widgets['graph_frame']))
    calc_button.grid(row=4, column=0, pady=2)

    save_button = ttk.Button(conf_widgets['frame'], text="Guardar Resultados", style="stBttn.TButton",
                           command=lambda: save_results(
                               conf_widgets['results'], 
                               "RESULTADOS DEL INTERVALO DE CONFIANZA"))
    save_button.grid(row=4, column=1, pady=2)

    # Frame para la gráfica
    conf_widgets['graph_frame'] = tk.Frame(tab)
    conf_widgets['graph_frame'].grid(row=1, column=0, sticky="nsew", padx=10, pady=2)

    return conf_widgets

def setup_hypothesis_test_tab(tab):
    """Configura la pestaña de pruebas de hipótesis"""
    # Variables para almacenar los widgets que necesitarán ser accedidos
    hypo_widgets = {}

    # Crear un estilo personalizado para las etiquetas
    estilo_labels = ttk.Style()
    estilo_labels.configure("stLabs.TLabel", 
                        font=("Arial", 12),
                        foreground="#333333",
                        background="#ccc5b9",
                        padding=(10, 5))

    # Crear un estilo personalizado para los combobox
    estilos_combo = ttk.Style()
    estilos_combo.configure("stCombos.TCombobox",
                        foreground="#333333",
                        background="#ccc5b9",
                        padding=(10, 5))

    # Crear un estilo personalizado para los entry
    estilo_entry = {
        "font": ("Arial", 12),
        "fg": "#333333",  # color del texto
        "insertbackground": "#333333",  # color del cursor
        "relief": "flat",  # tipo de borde (solid, raised, sunken, ridge, groove, flat)
        "borderwidth": 10,
        "highlightthickness": 1,
        "highlightbackground": "#FFFFFF",  # Color del borde igual al fondo
        "highlightcolor": "#ccc5b9"  # Color del resaltado igual al fondo
    }

    # Crear un estilo para el botón
    estilo_boton = ttk.Style()
    estilo_boton.configure("stBttn.TButton",
                foreground="#FFFFFF",
                font=("Arial", 12),
                justify="center",
                padding=(10, 5),
                relief= "raised",
                background="#197278",)
    
    # Configurar el grid para que se expanda
    tab.grid_rowconfigure(0, weight=1, minsize=300)
    tab.grid_rowconfigure(1, weight=1, minsize=350)
    
    tab.grid_columnconfigure(0, weight=1)
    
    # Contenedor para los widgets de entrada y salida de datos
    hypo_widgets['frame'] = tk.Frame(tab, bg="#ccc5b9")
    hypo_widgets['frame'].grid(row=0, column=0, sticky="ew", padx=10, pady=5)
    
    hypo_widgets['frame'].grid_rowconfigure(0, weight=3)
    hypo_widgets['frame'].grid_rowconfigure(1, weight=1)
    hypo_widgets['frame'].grid_rowconfigure(2, weight=1)
    hypo_widgets['frame'].grid_rowconfigure(3, weight=1)
    hypo_widgets['frame'].grid_rowconfigure(4, weight=1)
    hypo_widgets['frame'].grid_rowconfigure(5, weight=1)
    hypo_widgets['frame'].grid_rowconfigure(6, weight=1)
    
    hypo_widgets['frame'].grid_columnconfigure(0, weight=1)
    hypo_widgets['frame'].grid_columnconfigure(1, weight=1)
    hypo_widgets['frame'].grid_columnconfigure(2, weight=1)
    hypo_widgets['frame'].grid_columnconfigure(3, weight=5)

    # Etiqueta y campo para los datos de muestra
    ttk.Label(hypo_widgets['frame'], text="Datos de la muestra (separados por comas):", style="stLabs.TLabel").grid(row=1, column=0, sticky="w", padx=10)
    hypo_widgets['data_entry'] = tk.Entry(hypo_widgets['frame'], **estilo_entry)
    hypo_widgets['data_entry'].grid(row=1, column=1, columnspan=1, sticky="ew")

    # Hipótesis nula (media hipotética)
    ttk.Label(hypo_widgets['frame'], text="Valor de la hipótesis nula (μ₀):", style="stLabs.TLabel").grid(row=2, column=0, sticky="w", padx=10)
    hypo_widgets['null_hypothesis_entry'] = tk.Entry(hypo_widgets['frame'], **estilo_entry)
    hypo_widgets['null_hypothesis_entry'].insert(0, "0")
    hypo_widgets['null_hypothesis_entry'].grid(row=2, column=1, columnspan=2, sticky="ew")

    custom_font = tkfont.Font(family="Arial", size=12)

    # Tipo de prueba
    ttk.Label(hypo_widgets['frame'], text="Tipo de prueba:", style="stLabs.TLabel").grid(row=3, column=0, sticky="w", padx=10)
    hypo_widgets['test_type'] = ttk.Combobox(hypo_widgets['frame'], values=["Z (muestra grande o varianza conocida)", "t (muestra pequeña)"], style="stCombos.TCombobox", font=custom_font)
    hypo_widgets['test_type'].current(1)  # Seleccionar prueba t por defecto
    hypo_widgets['test_type'].grid(row=3, column=1, columnspan=2, sticky="ew")

    # Dirección de la prueba
    ttk.Label(hypo_widgets['frame'], text="Dirección de la prueba:", style="stLabs.TLabel").grid(row=4, column=0, sticky="w", padx=10)
    hypo_widgets['test_direction'] = ttk.Combobox(hypo_widgets['frame'], values=["Dos colas", "Cola izquierda", "Cola derecha"], style="stCombos.TCombobox", font=custom_font)
    hypo_widgets['test_direction'].current(0)  # Seleccionar dos colas por defecto
    hypo_widgets['test_direction'].grid(row=4, column=1, columnspan=2, sticky="ew")

    # Nivel de significancia
    ttk.Label(hypo_widgets['frame'], text="Nivel de significancia (α):", style="stLabs.TLabel").grid(row=5, column=0, sticky="w", padx=10)
    hypo_widgets['alpha_entry'] = tk.Entry(hypo_widgets['frame'], **estilo_entry)
    hypo_widgets['alpha_entry'].insert(0, "0.05")
    hypo_widgets['alpha_entry'].grid(row=5, column=1, columnspan=2, sticky="ew")

    # Área de resultados
    fuente_personalizada = tkfont.Font(family="Arial", size=12)
    ttk.Label(hypo_widgets['frame'], text="Resultados:", style="stLabs.TLabel").grid(row=1, column=3, sticky="nsew", padx=5)
    hypo_widgets['results'] = scrolledtext.ScrolledText(hypo_widgets['frame'], font=fuente_personalizada)
    hypo_widgets['results'].grid(row=2, rowspan=5, column=3, padx=5)

    # Botones
    load_button = tk.Button(hypo_widgets['frame'], text="  ⬆  ", font=('Arial', 12), command=lambda: load_data(hypo_widgets['frame'], None, hypo_widgets['data_entry']))
    load_button.config(background="#197278", foreground="#FFFFFF", relief="raised")
    load_button.grid(row=1, column=2, columnspan=1, sticky="nsew")

    calc_button = ttk.Button(hypo_widgets['frame'], text="  Calcular  ", style="stBttn.TButton",
                           command=lambda: calculate_hypothesis_test(
                               hypo_widgets['data_entry'], 
                               hypo_widgets['null_hypothesis_entry'], 
                               hypo_widgets['alpha_entry'], 
                               hypo_widgets['test_type'], 
                               hypo_widgets['test_direction'], 
                               hypo_widgets['results'], 
                               hypo_widgets['graph_frame']))
    calc_button.grid(row=6, column=0)

    save_button = ttk.Button(hypo_widgets['frame'], text="  Guardar Resultados  ", style="stBttn.TButton",
                           command=lambda: save_results(
                               hypo_widgets['results'], 
                               "RESULTADOS DE LA PRUEBA DE HIPÓTESIS"))
    save_button.grid(row=6, column=1)
        
    # Frame para la gráfica
    hypo_widgets['graph_frame'] = tk.Frame(tab)
    hypo_widgets['graph_frame'].grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    
    return hypo_widgets


def setup_help_tab(tab):
    """Configura la pestaña de ayuda"""
    help_text = """
    <h1 style="color: #eb5e28;">CALCULADORA ESTADÍSTICA 📊</h1>
    
    <p style="text-align: justify;">Bienvenido a la Calculadora Estadística, una herramienta diseñada para facilitar la realización de cálculos estadísticos fundamentales de manera intuitiva y eficiente. Esta aplicación es ideal tanto para estudiantes como para profesionales que necesitan realizar análisis estadísticos precisos.</p>
    
    <h2 style="color: #eb5e28;">1. INTERVALOS DE CONFIANZA 📏</h2>
    <p style="text-align: justify;">Un intervalo de confianza proporciona un rango de valores que probablemente contiene el parámetro poblacional desconocido.</p>
    <ul style="text-align: justify;">
        <li><b>Datos de muestra:</b> Ingresar los valores separados por comas o cargar desde un archivo.</li>
        <li><b>Nivel de confianza:</b> Típicamente 95% o 99%, representa la probabilidad de que el intervalo contenga el parámetro.</li>
        <li><b>Tipo de prueba:</b>
            <ul>
                <li><b>Z:</b> Para muestras grandes (n ≥ 30) o cuando se conoce la desviación estándar poblacional.</li>
                <li><b>t:</b> Para muestras pequeñas (n < 30) cuando no se conoce la desviación estándar poblacional.</li>
            </ul>
        </li>
    </ul>
    <p style="text-align: justify;"><b>Fórmula general:</b> x̄ ± (valor crítico) × (error estándar)</p>
    
    <h2 style="color: #eb5e28;">2. PRUEBAS DE HIPÓTESIS (PRUEBAS DE MEDIAS) 🧪</h2>
    <p style="text-align: justify;">Permiten tomar decisiones sobre parámetros poblacionales basadas en información muestral.</p>
    <ul style="text-align: justify;">
        <li><b>Datos de muestra:</b> Valores separados por comas o desde archivo.</li>
        <li><b>Hipótesis nula (μ₀):</b> Valor que se asume verdadero hasta que la evidencia indique lo contrario.</li>
        <li><b>Tipo de prueba:</b> Z o t (igual que para intervalos).</li>
        <li><b>Dirección de la prueba:</b>
            <ul>
                <li><b>Dos colas:</b> H₀: μ = μ₀ vs H₁: μ ≠ μ₀</li>
                <li><b>Cola izquierda:</b> H₀: μ ≥ μ₀ vs H₁: μ < μ₀</li>
                <li><b>Cola derecha:</b> H₀: μ ≤ μ₀ vs H₁: μ > μ₀</li>
            </ul>
        </li>
    </ul>
    <p style="text-align: justify;">La aplicación calcula:</p>
    <ul style="text-align: justify;">
        <li>Estadístico de prueba (Z o t)</li>
        <li>Valor p (probabilidad de obtener un resultado al menos tan extremo como el observado)</li>
        <li>Decisión (rechazar o no rechazar H₀)</li>
    </ul>
    
    <h2 style="color: #eb5e28;">3. CARGA Y GUARDADO DE DATOS 💾</h2>
    <ul style="text-align: justify;">
        <li><b>Cargar desde archivo:</b> Permite importar datos desde archivos CSV, Excel (.xlsx) o Parquet.</li>
        <li><b>Guardar resultados:</b> Exporta los resultados a un archivo de texto para uso posterior.</li>
    </ul>

    <h2 style="color: #eb5e28;">4. FINALIDAD DE LA APLICACIÓN 🎯</h2>
    <p style="text-align: justify;">La finalidad de esta aplicación es proporcionar una herramienta accesible y fácil de usar para realizar cálculos estadísticos esenciales. Con una interfaz amigable y funciones claras, esta calculadora estadística está diseñada para ayudar a los usuarios a obtener resultados precisos y confiables, facilitando así el análisis de datos en diversas áreas como la investigación, la educación y el análisis de negocios.</p>
    
    <p style="text-align: justify;"><b>NOTA:</b> Para resultados precisos, asegúrese de que los datos sean numéricos y que la muestra sea adecuada para el tipo de prueba seleccionado.</p>
    """
    
    # Crear el HTMLLabel con el texto de ayuda en HTML
    help_area = HTMLLabel(tab, html=help_text, width=80, height=30)
    help_area.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    
    # Configurar el grid para que se expanda
    tab.grid_rowconfigure(0, weight=1)
    tab.grid_columnconfigure(0, weight=1)
        
def main():
    # Crear la ventana principal
    ventana = tk.Tk()
    ventana.title("Calculadora estadística")
    ventana.config(bg="#fffcf2")
    ventana.attributes("-fullscreen", True)
    
    # Crear notebook (pestañas)
    notebook = ttk.Notebook(ventana)
    notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    
    # Configurar el grid para que se expanda
    ventana.grid_rowconfigure(0, weight=1)
    ventana.grid_rowconfigure(1, weight=1)
    ventana.grid_rowconfigure(2, weight=1)  
    
    ventana.grid_columnconfigure(0, weight=1)
    
    # Crear un estilo personalizado
    estiloTabs = ttk.Style()
    
    estilosHeaderTabs = ttk.Style()
    estilosHeaderTabs.theme_create( "headerTabs", parent="alt", settings={
        "TNotebook": {"configure": {"tabmargins": [2, 5, 2, 0], "background": "#fffcf2" }},
        "TNotebook.Tab": {
            "configure": {"padding": [5, 1], "background": "#f08c00", "font": ("Arial", 12) },
            "map":       {"background": [("selected", "#eb5e28")]},
    } } )
    estilosHeaderTabs.theme_use("headerTabs")

    # Configurar colores para cada pestaña
    estiloTabs.configure("tabs.TFrame", background="#ccc5b9")
    
    # Crear las pestañas
    conf_interval_tab = ttk.Frame(notebook, style="tabs.TFrame")
    hypothesis_test_tab = ttk.Frame(notebook, style="tabs.TFrame")
    help_tab = ttk.Frame(notebook, style="tabs.TFrame")
    
    notebook.add(conf_interval_tab, text="Intervalos de Confianza")
    notebook.add(hypothesis_test_tab, text="Pruebas de Medias")
    notebook.add(help_tab, text="Ayuda")
    
    # Configurar las pestañas
    setup_confidence_interval_tab(conf_interval_tab) 
    setup_hypothesis_test_tab(hypothesis_test_tab)
    setup_help_tab(help_tab)
    
    ventana.option_add('*TCombobox*Listbox.Background', '#fab005') # Color del fondo del menú
    ventana.option_add('*TCombobox*Listbox.selectBackground', '#f08c00') # Fondo de la opción seleccionada
    ventana.option_add('*TCombobox*Listbox.selectForeground', '#FFFFFF') # Texto de la opción seleccionada
    ventana.option_add('*TCombobox*Listbox.Font', ('Arial', 11, 'italic')) # Cambia la fuente y el tamaño
    
    # Iniciar el bucle principal
    ventana.mainloop()

if __name__ == "__main__":
    main()