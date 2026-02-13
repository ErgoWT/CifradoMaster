# IMPORTS
import json
import time
import ssl
from pathlib import Path
# ============================================
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import cycler
from PIL import Image

# ========== CONFIGURACION MQTT ==========
BROKER = "raspberrypiJED.local"
PORT = 8883
QOS = 1
USERNAME = "usuario1"
PASSWORD = "qwerty123"
TOPIC_KEYS = "graficas/keys"
TOPIC_DATA = "graficas/data"
CA_CERT_PATH = "/etc/mosquitto/ca_certificates/ca.crt"
# ========================================

# ========== PARAMETROS DE CIFRADO ==========
# Parámetros para Rössler
ROSSLER_PARAMS = {
    "a": 0.2,
    "b": 0.2,
    "c": 5.7,
}
TIEMPO_SINC = 6000 # Tiempo de sincronización experimental
H = 0.01 # Paso de integración
Y0 = [0.1, 0.1, 0.1] # Condiciones iniciales del sistema de Rössler
KEYSTREAM = 30000
IMG_SCALE = 0.02
# Parámetros para Logistic Map
LOGISTIC_PARAMS = {
    "aLog": 3.99,
    "x0_log": 0.4
}

# ========== RUTAS Y ARCHIVOS ==========
CARPETA_CIFRADO = Path("Graficas_Maestro")
CARPETA_CIFRADO.mkdir(parents=True, exist_ok=True)
IMAGEN_ENTRADA = Path("Prueba2.jpg")
RUTA_IMAGEN_CIFRADA = CARPETA_CIFRADO / "ImagenCifrada_TLS.png"
RUTA_TIMINGS = CARPETA_CIFRADO / "tiempos_procesos.csv"
RUTA_DISPERSION = CARPETA_CIFRADO / "diagrama_dispersion.png"
RUTA_SERIES_VECTORES = CARPETA_CIFRADO / "series_difusion_logistico_rossler.png"
RUTA_VECTOR_CIFRADO_SERIE = CARPETA_CIFRADO / "vector_cifrado.png"

# ========== CONSTANTES AUXILIARES ==========
PUNTOS_EVAL = 15000  # Número de puntos a evaluar en las gráficas


def set_mpl_style_journal():
    # Paleta Okabe–Ito (colorblind-friendly) + tono sobrio
    # Fuente: ampliamente usada en artículos por su accesibilidad y contraste.
    colorblind_palette = [
        "#000000",
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        "#D55E00",  # vermillion        
        "#CC79A7",  # reddish purple
        "#56B4E9",  # sky blue
        "#E69F00",  # orange
        
    ]

    plt.rcParams.update({
        # ---------------------------
        # Tipografía
        # ---------------------------
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],  # fallback seguro
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # ---------------------------
        # Ejes y ticks (look “clean”)
        # ---------------------------
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,

        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # ---------------------------
        # Líneas y ciclo de colores
        # ---------------------------
        "lines.linewidth": 0.8,
        "lines.markersize": 4.0,
        "axes.prop_cycle": cycler(color=colorblind_palette),

        # Colormap por defecto (para imshow, scatter con cmap, etc.)
        "image.cmap": "viridis",

        # ---------------------------
        # Grid: apagado por defecto, pero bonito si lo activas
        # ---------------------------
        "axes.grid": False,
        "grid.color": "0.6",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,

        # ---------------------------
        # Exportación / calidad
        # ---------------------------
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "pdf",      # preferente
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        # Tipos de fuente embebidos (mejor compatibilidad en PDF/EPS)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # ---------------------------
        # Leyenda
        # ---------------------------
        "legend.frameon": False,
    })






# ========== SISTEMA DE RÖSSLER ==========
def rossler_maestro(t, state, a, b, c):
    # Ecuaciones del sistema de Rössler
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

# ========== FUNCIÓN DE DIFUSIÓN ==========
def aplicar_difusion(vector_inf, nmax):
    """
    Aquí aplicamos la etapa de difusón a la imagen utilizando el mapa logístico
    Parámetros:
    ----------
    imagen : PIL.Image
        Es una imagen RGB, que se va a cifrar
    a_log : float
        Parámetro de control del mapa logístico (3.57 < a_log <= 4) para generar caos
    x0_log : float
        Condición inicial del mapa logístico (0 < x0_log < 1)
    
    Retorna:
    -------
    difusion : np.ndarray
        Vector 1D normalizado en [0, 1] con la imagen difundida
    vector_logistico : np.ndarray
        Secuencia del mapa logístico usado como máscara de difusión
    """
    # Resumen:
    # 1) Se genera una secuencia logística
    # 2) Se generan posiciones de mezcla a partir de la secuencia logística
    # 3) Se aplica la difusión permutando los píxeles de la imagen según las posiciones generadas
    

    print("[DIFUSION] INICIANDO...")
    t_inicio_difusion = time.perf_counter()

    # 1. Generamos el vector logístico
    vector_logistico = np.zeros(nmax)
    x = LOGISTIC_PARAMS['x0_log']
    for i in range(nmax):
        x = LOGISTIC_PARAMS['aLog'] * x * (1 - x)
        vector_logistico[i] = x

    # 2. Generamos el vector mezcla (posiciones)
    vector_mezcla = np.floor(vector_logistico * nmax).astype(int)
    # Entrada:
    # vector_logistico: array de valores entre 0 y 1 generados con el mapa logístico
    # nmax: tamaño del vector de la imagen, es decir, el número de píxeles multiplicado por 3 (RGB)
    # Transformacion:
    # Se escala cada valor del vector_logistico al rango [0, nmax - 1]
    # multiplicando por (nmax-1) y luego aplicando floor para obtener enteros (redondear hacia abajo
    # y evitar valores fuera de rango).
    # Se convierte a enteros con astype(int) para asegurar que son índices válidos.
    # 
    # Cada valor en vector_mezcla es un índice entre 0 y nmax-1
    # Por ejemplo:
    # vector_logistico = [0.123, 0.757, 0.432, ...] Valores entre 0 y 1
    # nmax = 1000 (tamaño del vector de la imagen)
    # vector_mezcla = [123, 756 (se redondeó hacia abajo), 432, ...] Índices enteros entre 0 y 999 


    # 3. Aplicamos permutación con marcador 260
    vector_temp = vector_inf.copy() # Copiamos el vector original
    difusion = np.zeros(nmax) # Creamos un vector de difusión vacío para almacenar los resultados
    contador = 0 # Contador para el número de asignaciones realizadas

    # 3.1 Primera pasada: asignación desde posiciones aleatorias
    # Iteramos sobre el vector_mezcla para asignar valores desde posiciones aleatorias
    # Para cada posicion en pos:
    #   Si el valor en vector_temp[pos] no es 260.0, lo asignamos a difusion
    #   y marcamos vector_temp[pos] como 260.0 (usado)    
    for i in range(nmax):
        pos = vector_mezcla[i]
        if vector_temp[pos] != 260.0:
            difusion[contador] = vector_temp[pos]
            contador += 1
            vector_temp[pos] = 260.0  # Marcamos como usado
    # Ejemplo:
    # vector_temp = [0.1, 0.4, 0.7, 0.3, 0.9, ...] (valores normalizados)
    # vector_mezcla = [123, 756, 432, ...]
    # Iteracion 1 (pos=123):
    #   Si vector_temp[123] != 260.0 (por ejemplo, 0.5), entonces:
    #     difusion[0] = 0.5 (asignamos el valor)
    #     vector_temp[123] = 260.0 (marcamos como usado)
    #   Contador incrementa a 1
    # Iteracion 2 (pos=756):
    #   Si vector_temp[756] != 260.0 (por ejemplo, 0.8), entonces:
    #     difusion[1] = 0.8 (asignamos el valor)
    #     vector_temp[756] = 260.0 (marcamos como usado)
    #   Contador incrementa a 2
    # Continuamos hasta que hayamos asignado nmax valores o no queden valores disponibles

    # 3.2 Segunda pasada: asignación de los restantes
    for j in range(nmax):
        if contador >= nmax:
            break
        if vector_temp[j] != 260.0:
            difusion[contador] = vector_temp[j]
            contador += 1
    # Iteramos secuencialmente por todas las posiciones
    # Para cada posicion j:
    #   Si el valor en vector_temp[j] no es 260.0, lo asignamos a difusion
    #   y marcamos vector_temp[j] como 260.0 (usado)
    # Ejemplo:
    # vector_temp = [0.1, 0.4, 0.7, 0.3, 0.9, ...] (valores normalizados)
    # Iteracion 1 (j=0):
    #   Si vector_temp[0] != 260.0 (por ejemplo, 0.1), entonces:
    #     difusion[contador] = 0.1 (asignamos el valor)
    #     vector_temp[0] = 260.0 (marcamos como usado)
    #   Contador incrementa

    t_fin_difusion = time.perf_counter()
    tiempo_difusion = t_fin_difusion - t_inicio_difusion
    print(f"[DIFUSION] Tiempo de difusión: {tiempo_difusion:.4f} segundos")
    print("[DIFUSION] DIFUSIÓN COMPLETADA")
    

    return difusion, vector_logistico, tiempo_difusion

def aplicar_confusion(difusion, vector_logistico, nmax, rosslerParams):
    """
    Se aplica la etapa de confusión utilizando el oscilador de Rössler
    Parámetros:
    ----------
    difusion : np.ndarray
        Vector 1D normalizado en [0, 1] con la imagen difundida
    vector_logistico : np.ndarray
        Secuencia del mapa logístico usado como máscara de difusión (se suma en confusión)
    nmax : int
        Número total de elementos en el vector de la imagen (ancho * alto * 3)
    rosslerParams : dict
        Parámetros del sistema de Rössler (a, b, c)
    Retorna:
    -------
    vector_cifrado : np.ndarray
        Vector 1D normalizado en [0, 1] con la imagen cifrada, resultado de sumar difusión,
        secuencia logística y la señal x del sistema de Rössler
    y_sinc : np.ndarray
        Trayectoria y del sistema de Rössler (usada para sincronización)
    t : np.ndarray
        Vector de tiempos correspondiente a las trayectorias del sistema de Rössler
    """

    # Resumen:
    # 1) Se resuelve el sistema de Rössler para obtener las trayectorias
    # 2) Tras un tiempo de sincronización, se extrae la señal x para confusión
    # 3) Esa señal se redimensiona a nmax y se suma a la difusión y la secuencia logística
    #    para obtener el vector cifrado final
    
    print("[CONFUSION] APLICANDO CONFUSIÓN...")
    t_inicio_confusion = time.perf_counter()

    # 1. Se calculan las iteraciones totales (sincronización + cifrado)
    iteraciones = TIEMPO_SINC + KEYSTREAM
    print(f"[CONFUSION] Iteraciones totales: {iteraciones}")

    # 2. Resolver el sistema de Rössler
    t_span = (0, iteraciones * H)
    t_eval = np.linspace(0, iteraciones * H, iteraciones) # linspace es una funcion que genera un array de valores equiespaciados entre dos puntos, a
    # diferencia de arange, donde arange genera valores con un paso fijo.
    t_inicio_rossler = time.perf_counter()
    solucion = solve_ivp(
        fun = rossler_maestro,
        y0 = Y0,
        args = tuple(rosslerParams.values()),
        t_span = t_span,
        t_eval = t_eval,
        method = 'RK23',
        rtol = 1e-6,
        atol = 1e-8
    )
    t_fin_rossler = time.perf_counter()
    tiempo_rossler = t_fin_rossler - t_inicio_rossler

    # 3. Extraer las trayectorias del sistema de Rössler
    x = solucion.y[0] # Señal x completa
    x_key = solucion.y[0][TIEMPO_SINC:] # Para la confusion
    y = solucion.y[1] # Para sincronizacion
    z = solucion.y[2]
    t = solucion.t

    x_cif = np.resize(x_key, nmax)

    # 4. Aplicar confusión (solo después del tiempo de sincronización)
    vector_cifrado = np.zeros(nmax)
    vector_cifrado = difusion + vector_logistico + x_cif
    print("[CONFUSION] Confusión aplicada correctamente")

    t_fin_confusion = time.perf_counter()
    tiempo_confusion = t_fin_confusion - t_inicio_confusion

    print(f"[CONFUSION] Tiempo de integración de Rössler: {tiempo_rossler:.4f} segundos")
    print(f"[CONFUSION] Tiempo total de confusión: {tiempo_confusion:.4f} segundos")

    return vector_cifrado, x, x_key, y, z, t, tiempo_rossler, tiempo_confusion

def cargar_imagen():
    """
    Cargamos la imagen de entrada definida en IMAGEN_ENTRADA

    Retorna:
    -------
    imagen : PIL.Image
        Imagen cargada desde el archivo, siendo esta RGB que se va a cifrar
    """
    imagen = Image.open(IMAGEN_ENTRADA)
    vector_inf = np.array(imagen)
    alto, ancho, canales = vector_inf.shape
    vector_inf = vector_inf.flatten().astype(np.float64)/255.0
    nmax = vector_inf.size
    print("[CARGA] Imagen cargada y vectorizada correctamente")
    return imagen, vector_inf, ancho, alto, nmax

def preparar_payload(vector_cifrado, x_master, y_maestro, z_master, t_maestro, ancho, alto, nmax):
    """
    Se prepara el diccionario de datos para envíar mediante MQTT

    Retorna:
    -------
    data : dict
        Diccionario con los datos necesarios para el descifrado
    """

    data = {
        "vector_cifrado": vector_cifrado.tolist(),
        "x_maestro": x_master.tolist(),
        "y_maestro": y_maestro.tolist(),
        "z_maestro": z_master.tolist(),
        "t_maestro": t_maestro.tolist(),
        "ancho": ancho,
        "alto": alto,
        "nmax": nmax,
        "tiempo_sinc": TIEMPO_SINC,
        "KEYSTREAM": KEYSTREAM
    }
    return data

def graficas(imagen, difusion, vector_cifrado, ancho, alto, ruta_salida=RUTA_IMAGEN_CIFRADA, orden="F"):
    """
    Figura comparativa (3 paneles):
      (a) Original
      (b) Después de difusión (pseudoimagen)
      (c) Después de confusión (pseudoimagen normalizada)

    - 'orden' debe coincidir con el usado al vectorizar (flatten).
      Con tu cargar_imagen() actual, lo correcto es orden="C".
    - Guarda salida principal en PDF y un PNG de preview si ruta_salida termina en .png/.jpg/...
    """
    # --- Normalizar entradas ---
    difusion = np.asarray(difusion).ravel()
    vector_cifrado = np.asarray(vector_cifrado).ravel()

    n_esperado = int(alto) * int(ancho) * 3
    if difusion.size != n_esperado:
        raise ValueError(f"[GRAFICAS] 'difusion' tiene {difusion.size} elementos, pero se esperaban {n_esperado} (alto*ancho*3).")
    if vector_cifrado.size != n_esperado:
        raise ValueError(f"[GRAFICAS] 'vector_cifrado' tiene {vector_cifrado.size} elementos, pero se esperaban {n_esperado} (alto*ancho*3).")

    # --- (b) Pseudoimagen de difusión ---
    # Difusión se asume en [0,1], pero por seguridad recortamos
    dif_img = np.clip(difusion, 0.0, 1.0)
    dif_img = np.rint(dif_img * 255.0).astype(np.uint8).reshape((alto, ancho, 3), order=orden)

    # --- (c) Pseudoimagen cifrada (confusión) ---
    # Para visualizar, normalizamos min-max (evita reventar el rango en pantalla)
    vmin = float(np.min(vector_cifrado))
    vmax = float(np.max(vector_cifrado))
    if np.isclose(vmax, vmin):
        cif_norm = np.zeros_like(vector_cifrado, dtype=np.float64)
    else:
        cif_norm = (vector_cifrado - vmin) / (vmax - vmin)

    cif_img = np.clip(cif_norm, 0.0, 1.0)
    cif_img = np.rint(cif_img * 255.0).astype(np.uint8).reshape((alto, ancho, 3), order=orden)

    # --- Figura estilo “paper” ---
    fig, axes = plt.subplots(
        nrows=1, ncols=3,
        figsize=(7.2, 2.6),
        constrained_layout=True
    )

    paneles = [
        ("(a)", "Original", np.asarray(imagen), None),
        ("(b)", "Después de Difusión", dif_img, None),
        ("(c)", "Después de Confusión", cif_img, None),
    ]

    for ax, (tag, titulo, img, cmap) in zip(axes, paneles):
        ax.imshow(img, interpolation="nearest", cmap=cmap)
        ax.set_title(titulo, pad=6)
        ax.axis("off")
        ax.text(
            0.01, 0.99, tag,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=10
        )

    # --- Guardado: PDF principal + preview opcional ---
    ruta_salida = Path(ruta_salida)
    suffix = ruta_salida.suffix.lower()

    # PDF “de revista”
    ruta_pdf = ruta_salida if suffix == ".pdf" else ruta_salida.with_suffix(".pdf")
    fig.savefig(ruta_pdf)

    # Preview raster (si tu ruta era png/jpg/etc.)
    if suffix in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        fig.savefig(ruta_salida)

    plt.close(fig)

    print(f"[GRAFICA] Comparativa guardada (principal): {ruta_pdf}")
    if suffix in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        print(f"[GRAFICA] Preview guardado: {ruta_salida}")


def graficar_series_vectores(
    difusion,
    vector_logistico,
    x_conf,
    puntos_eval=PUNTOS_EVAL,
    ruta_salida=RUTA_SERIES_VECTORES,
    show_grid=False
):
    """
    Grafica (en 3 subplots) las series usadas en el cifrado:
      1) Vector de difusión
      2) Vector logístico
      3) Serie x del Rössler usada para confusión

    - Respeta el estilo global (rcParams).
    - Guarda salida principal en PDF (vectorial) y un PNG de preview si aplica.
    - Recorta automáticamente para evitar errores si las series son más cortas que puntos_eval.

    Parámetros:
        difusion, vector_logistico, x_conf: array-like
        puntos_eval (int): máximo de puntos a mostrar
        ruta_salida (Path o str): ruta base de guardado (puede terminar en .png)
        show_grid (bool): si True, activa grid con el estilo global (tenue y discontinuo).
    """

    # ---- Normalización básica de entradas (sin alterar tus datos originales) ----
    difusion = np.asarray(difusion).ravel()
    vector_logistico = np.asarray(vector_logistico).ravel()
    x_conf = np.asarray(x_conf).ravel()

    # Para compartir eje X sin broncas, usamos el mismo N para las 3 series
    n = min(int(puntos_eval), difusion.size, vector_logistico.size, x_conf.size)
    if n <= 0:
        raise ValueError("No hay datos suficientes para graficar (n <= 0).")

    # ---- Figura tipo “paper”: compacta, limpia, consistente ----
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        figsize=(6.5, 6.0),          # tamaño práctico para reporte/paper
        constrained_layout=True
    )

    series = [
        (difusion[:n], "Difusión", "(a)"),
        (vector_logistico[:n], "Logístico", "(b)"),
        (x_conf[:n], "x (Rössler)", "(c)"),
    ]

    for ax, (y, ylabel, panel) in zip(axes, series):
        ax.plot(y)  # usa el ciclo de colores global (colorblind-friendly)
        ax.set_ylabel(ylabel)

        # Etiqueta tipo “panel” (a), (b), (c) en esquina superior izquierda
        ax.text(
            0.01, 0.92, panel,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left"
        )

        # Grid: apagado por defecto, si se activa usa tu estilo global (tenue/discontinuo)
        ax.grid(show_grid)

    axes[-1].set_xlabel("Índice (muestra)")

    # Si quieres un título general (sobrio), puedes dejarlo:
    # fig.suptitle("Series utilizadas para difusión y confusión", y=1.02)

    # ---- Guardado profesional: PDF principal + PNG opcional ----
    ruta_salida = Path(ruta_salida)

    # PDF (vectorial) como salida “de revista”
    ruta_pdf = ruta_salida.with_suffix(".pdf")
    fig.savefig(ruta_pdf)

    # PNG de preview (solo si tu ruta original era png/jpg/etc.)
    # Esto te sirve para ver rápido resultados sin abrir el PDF.
    if ruta_salida.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        fig.savefig(ruta_salida)

    plt.close(fig)

    print(f"[GRAFICA] Series guardadas (principal): {ruta_pdf}")
    if ruta_salida.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        print(f"[GRAFICA] Preview guardado: {ruta_salida}")

def graficar_vector_cifrado(vector_cifrado, show_grid=False):
    """
    Grafica el vector cifrado como señal 1D con estilo tipo revista (usa rcParams globales).
    
    - Grid apagado por defecto (show_grid=False).
    - Guarda salida principal en PDF (vectorial) y preview en PNG si tu ruta lo indica.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Asegurar vector 1D
    y = np.asarray(vector_cifrado).ravel()

    # Longitud segura para graficar
    n = min(int(PUNTOS_EVAL), y.size)
    if n <= 0:
        raise ValueError("No hay datos suficientes en vector_cifrado para graficar.")

    # Figura compacta tipo paper
    fig, ax = plt.subplots(
        figsize=(6.5, 2.6),
        constrained_layout=True
    )

    # Plot: usa ciclo de colores global + linewidth global
    ax.plot(y[:n])

    ax.set_xlabel("Muestras")
    ax.set_ylabel("Valor")
    ax.set_title("Vector cifrado")

    # Grid: solo si lo pides; si se activa respeta tu estilo global (tenue/discontinuo)
    ax.grid(show_grid)

    # Guardado: PDF principal + PNG preview
    ruta_salida = Path(RUTA_VECTOR_CIFRADO_SERIE)

    ruta_pdf = ruta_salida.with_suffix(".pdf")
    fig.savefig(ruta_pdf)

    # Si tu ruta tiene extensión raster, guardamos también el preview
    if ruta_salida.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        fig.savefig(ruta_salida)

    plt.close(fig)

    print(f"[GRAFICA] Vector cifrado guardado (principal): {ruta_pdf}")
    if ruta_salida.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        print(f"[GRAFICA] Vector cifrado guardado (preview):   {ruta_salida}")

def graficar_dispersion(imagen, vector_cifrado, method="auto", max_scatter_points=200_000, show_grid=False):
    """
    Diagrama de dispersión / densidad entre imagen original y 'pseudo-imagen' cifrada.
    Ambas series se normalizan a [0, 1].

    method:
        - "auto": usa hist2d si hay muchos puntos; scatter si hay pocos
        - "scatter": scatter (con rasterizado para PDF ligero)
        - "hist2d": histograma 2D (recomendado para millones de puntos)

    max_scatter_points:
        Límite de puntos para scatter (si supera, auto usa hist2d o se submuestrea).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from matplotlib.colors import LogNorm

    # --- Original normalizada a [0,1] ---
    x = np.asarray(imagen, dtype=np.float32).ravel() / 255.0

    # --- Cifrado normalizado a [0,1] ---
    y_raw = np.asarray(vector_cifrado, dtype=np.float32).ravel()
    y = (y_raw - np.min(y_raw)) / (np.max(y_raw) - np.min(y_raw) + 1e-12)

    # Asegurar misma longitud
    n = min(x.size, y.size)
    if n <= 0:
        raise ValueError("No hay datos suficientes para graficar dispersión.")
    x = x[:n]
    y = y[:n]

    # Elegir método automáticamente
    if method not in ("auto", "scatter", "hist2d"):
        raise ValueError("method debe ser 'auto', 'scatter' o 'hist2d'.")

    if method == "auto":
        # Si hay demasiados puntos, densidad es más profesional + rápido
        method_use = "hist2d" if n > max_scatter_points else "scatter"
    else:
        method_use = method

    fig, ax = plt.subplots(figsize=(4.2, 4.2), constrained_layout=True)

    if method_use == "scatter":
        # Si sigue siendo grande, submuestreamos para no matar rendimiento
        if n > max_scatter_points:
            # Submuestreo determinista (uniforme) para reproducibilidad
            idx = np.linspace(0, n - 1, max_scatter_points, dtype=int)
            xs = x[idx]
            ys = y[idx]
        else:
            xs, ys = x, y

        ax.scatter(
            xs, ys,
            s=2, alpha=0.15,
            linewidths=0,
            rasterized=True  # clave: PDF liviano aunque sea scatter
        )

    else:
        # Histograma 2D: muy recomendado para muchos puntos
        bins = 200  # puedes subir a 300 si quieres más resolución
        H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])

        # Log para que se vea estructura aunque haya zonas súper densas
        # (vmin=1 evita log(0))
        im = ax.imshow(
            H.T,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="equal",
            interpolation="nearest",
            norm=LogNorm(vmin=1, vmax=max(1, H.max()))
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Conteo (escala log)")

    # Línea de referencia y=x (no debe tomar el color del ciclo)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8, color="0.35")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Original (normalizada)")
    ax.set_ylabel("Cifrada (normalizada)")
    ax.set_title("Relación píxel a píxel: original vs cifrada")

    # Grid opcional (si se activa, respeta el estilo global)
    ax.grid(show_grid)

    # --- Guardado: PDF principal + PNG preview ---
    ruta_salida = Path(RUTA_DISPERSION)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    ruta_pdf = ruta_salida.with_suffix(".pdf")
    fig.savefig(ruta_pdf)

    if ruta_salida.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        fig.savefig(ruta_salida)

    plt.close(fig)

    print(f"[GRAFICA] Dispersión guardada (principal): {ruta_pdf}")
    if ruta_salida.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        print(f"[GRAFICA] Dispersión guardada (preview):   {ruta_salida}")


def calcular_hamming(imagen, vector_cifrado, ancho, alto):
    """
    Se calcula la distancia Hamming entre la imagen original y la cifrada
    """
    # Imagen original en bytes (0-255)
    orig_img = np.array(imagen).astype(np.uint8)
    orig_flat = orig_img.flatten()

    # Reconstruir la pseudo-imagen cifrada en 0-255 (igual que en graficas)
    cifrado_norm = (vector_cifrado - np.min(vector_cifrado)) / (
        np.max(vector_cifrado) - np.min(vector_cifrado) + 1e-12
    )
    cifrado_img = (cifrado_norm * 255).reshape((alto, ancho, 3), order = 'F').astype(np.uint8)
    cifrado_flat = cifrado_img.flatten()

    # Asegurar misma longitud
    n_bytes = min(orig_flat.size, cifrado_flat.size)
    orig_flat = orig_flat[:n_bytes]
    cifrado_flat = cifrado_flat[:n_bytes]

    # Convertir a bits
    orig_bits = np.unpackbits(orig_flat)
    cifrado_bits = np.unpackbits(cifrado_flat)

    n_bits = min(orig_bits.size, cifrado_bits.size)
    orig_bits = orig_bits[:n_bits]
    cifrado_bits = cifrado_bits[:n_bits]

    # Distancia Hamming absoluta y normalizada
    hamming_abs = np.sum(orig_bits != cifrado_bits)
    hamming_norm = hamming_abs / n_bits

    print(f"[HAMMING] Distancia Hamming absoluta: {hamming_abs}")
    print(f"[HAMMING] Distancia Hamming normalizada: {hamming_norm:.6f}")


def registrar_tiempos(tiempo_difusion, tiempo_rossler, tiempo_confusion, tiempo_mqtt, tiempo_programa):
    """
    Se registran las métricas de tiempo para cada proceso en un archivo CSV
    """
    registro = {
        "timestamp": time.strftime("%m-%d %H:%M:%S"),
        "tiempo_difusion_segundos": tiempo_difusion,
        "tiempo_rossler_segundos": tiempo_rossler,
        "tiempo_confusion_segundos": tiempo_confusion,
        "tiempo_mqtt_segundos": tiempo_mqtt,
        "tiempo_programa_segundos": tiempo_programa
    }

    df = pd.DataFrame([registro])
    archivo = RUTA_TIMINGS.exists()
    df.to_csv(RUTA_TIMINGS, mode='a', index = False, header = not archivo)
    print(f"[TIEMPOS] Tiempos registrados en {RUTA_TIMINGS}")

def main():
    set_mpl_style_journal()
    inicio_programa = time.perf_counter()
    # 1. Cargar la imagen
    imagen, vector_inf, ancho, alto, nmax = cargar_imagen()
    # Mostrar medidas de la imagen
    print(f"[CARGA] Medidas de la imagen: Ancho={ancho}, Alto={alto}, Canales=3, Total píxeles={nmax//3}, Total valores={nmax}")

    # 2. Aplicar difusión
    difusion_x, vector_logistico, tiempo_difusion = aplicar_difusion(vector_inf, nmax)
    difusion = difusion_x * IMG_SCALE
    # 3. Aplicar confusión
    vector_cifrado, x, x_key, y_sinc, z, t, tiempo_rossler, tiempo_confusion = aplicar_confusion(difusion, vector_logistico, nmax, ROSSLER_PARAMS)

    # 4. Preparar datos para MQTT
    data = preparar_payload(vector_cifrado, x, y_sinc, z, t, ancho, alto, nmax)

    # 5. Publicar en MQTT con TLS
    t_inicio_mqtt = time.perf_counter()
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set(ca_certs=CA_CERT_PATH, tls_version=ssl.PROTOCOL_TLS_CLIENT)
    client.tls_insecure_set(False)
    client.connect(BROKER, PORT, 60)
    print("[MQTT] Conectado al broker MQTT con TLS")

    # Publicar parámetros keys por TLS
    client.publish(TOPIC_KEYS, json.dumps(
        {
            "ROSSLER_PARAMS": ROSSLER_PARAMS,
            "LOGISTIC_PARAMS": LOGISTIC_PARAMS
        }),
        qos=QOS,
        retain=True
    )
    time.sleep(0.5)

    client.publish(TOPIC_DATA, json.dumps(data), qos=QOS, retain = True)
    time.sleep(0.5)
    t_fin_mqtt = time.perf_counter()
    tiempo_mqtt = t_fin_mqtt - t_inicio_mqtt
    print(f"[MQTT] Tiempo de publicación MQTT con TLS: {tiempo_mqtt:.4f} segundos")
    client.disconnect()
    print("[MQTT] Datos publicados correctamente en MQTT")
    fin_programa = time.perf_counter()
    tiempo_programa = fin_programa - inicio_programa

    print(f"[PROGRAMA] Tiempo total del programa: {tiempo_programa:.4f} segundos")

    registrar_tiempos(
        tiempo_difusion,
        tiempo_rossler,
        tiempo_confusion,
        tiempo_mqtt,
        tiempo_programa
    )

    # 6. Generar gráficas
    graficas(imagen, difusion, vector_cifrado, ancho, alto)
    graficar_dispersion(imagen, vector_cifrado)
    calcular_hamming(imagen, vector_cifrado, ancho, alto)
    graficar_series_vectores(difusion, vector_logistico, x_key)
    graficar_vector_cifrado(vector_cifrado)

    print("[PROGRAMA] Proceso de cifrado completado")

if __name__ == "__main__":
    main()
