# IMPORTS
import json
import time
from pathlib import Path
# ============================================
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from PIL import Image

# ========== CONFIGURACION MQTT ==========
BROKER = "192.168.0.55"
PORT = 1883
QOS = 1
USERNAME = "usuario1"
PASSWORD = "qwerty123"
TOPIC_KEYS = "chaos1883/keys"
TOPIC_DATA = "chaos1883/data"
# ========================================

# ========== PARAMETROS DE CIFRADO ==========
# Parámetros para Rössler
ROSSLER_PARAMS = {
    "a": 0.2,
    "b": 0.2,
    "c": 5.7,
}
TIME_SINC = 2500 # Tiempo de sincronización experimental
H = 0.01 # Paso de integración
Y0 = [0.1, 0.1, 0.1] # Condiciones iniciales del sistema de Rössler
# Parámetros para Logistic Map
LOGISTIC_PARAMS = {
    "aLog": 3.99,
    "x0_log": 0.4
}

# ========== RUTAS Y ARCHIVOS ==========
CARPETA_CIFRADO = Path("Cifrado_1883_solo")
IMAGEN_ENTRADA = Path("Prueba.jpg")
RUTA_IMAGEN_CIFRADA = CARPETA_CIFRADO / "ImagenCifrada_1883_solo.png"
RUTA_TIMINGS = CARPETA_CIFRADO / "tiempos_procesos.csv"
RUTA_DISPERSION = CARPETA_CIFRADO / "diagrama_dispersion.png"
RUTA_HAMMING = CARPETA_CIFRADO / "hamming.png"

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
    iteraciones = TIME_SINC + nmax
    print(f"[CONFUSION] Iteraciones totales: {iteraciones}")

    # 2. Resolver el sistema de Rössler
    t_span = (0, iteraciones * H)
    t_eval = np.linspace(0, iteraciones * H, iteraciones)
    t_inicio_rossler = time.perf_counter()
    solucion = solve_ivp(
        fun = rossler_maestro,
        y0 = Y0,
        args = tuple(rosslerParams.values()),
        t_span = t_span,
        t_eval = t_eval,
        method = 'RK45',
        rtol = 1e-8,
        atol = 1e-8
    )
    t_fin_rossler = time.perf_counter()
    tiempo_rossler = t_fin_rossler - t_inicio_rossler

    # 3. Extraer las trayectorias del sistema de Rössler
    x = solucion.y[0][TIME_SINC:] # Para la confusion
    y = solucion.y[1] # Para sincronizacion
    t = solucion.t

    # 4. Aplicar confusión (solo después del tiempo de sincronización)
    vector_cifrado = np.zeros(nmax)
    vector_cifrado = difusion + vector_logistico + x
    print("[CONFUSION] Confusión aplicada correctamente")

    t_fin_confusion = time.perf_counter()
    tiempo_confusion = t_fin_confusion - t_inicio_confusion

    print(f"[CONFUSION] Tiempo de integración de Rössler: {tiempo_rossler:.4f} segundos")
    print(f"[CONFUSION] Tiempo total de confusión: {tiempo_confusion:.4f} segundos")

    return vector_cifrado, y, t, tiempo_rossler, tiempo_confusion

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
    vector_inf = vector_inf.flatten().astype(np.float32)/255.0
    nmax = vector_inf.size
    print("[CARGA] Imagen cargada y vectorizada correctamente")
    return imagen, vector_inf, ancho, alto, nmax

def preparar_payload(vector_cifrado, y_sinc, t, ancho, alto, nmax):
    """
    Se prepara el diccionario de datos para envíar mediante MQTT

    Retorna:
    -------
    data : dict
        Diccionario con los datos necesarios para el descifrado
    """

    data = {
        "vector_cifrado": vector_cifrado.tolist(),
        "y_sinc": y_sinc.tolist(),
        "times": t.tolist(),
        "ancho": ancho,
        "alto": alto,
        "nmax": nmax,
        "time_sinc": TIME_SINC,
    }
    return data

def graficas(imagen, difusion, vector_cifrado, ancho, alto):
    """
    Genera y guarda la figura comparativa:
    original, después de difusión y después de confusión.
    """
    plt.figure(figsize=(15, 5))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(imagen)
    plt.title("Original")
    plt.axis("off")

    # Después de difusión
    plt.subplot(1, 3, 2)
    difusion_img = (difusion * 255).reshape((alto, ancho, 3)).astype(np.uint8)
    plt.imshow(difusion_img)
    plt.title("Después de Difusión")
    plt.axis("off")

    # Después de confusión (pseudo-imagen)
    plt.subplot(1, 3, 3)
    cifrado_img = (
        (vector_cifrado - np.min(vector_cifrado))
        / (np.max(vector_cifrado) - np.min(vector_cifrado))
        * 255
    ).reshape((alto, ancho, 3)).astype(np.uint8)
    plt.imshow(cifrado_img)
    plt.title("Después de Confusión")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(RUTA_IMAGEN_CIFRADA)
    print(f"Resultados del proceso completo guardados en {RUTA_IMAGEN_CIFRADA}")

def graficar_dispersion(imagen, vector_cifrado):
    """
    Genera y guarda el diagrama de dispersión entre la imagen original y la cifrada
    Ambas están normalizadas
    """
    img_original = np.array(imagen).flatten().astype(np.float32)/255.0
    
    cifrado_normalizado = (vector_cifrado - np.min(vector_cifrado)) / (
        np.max(vector_cifrado) - np.min(vector_cifrado) + 1e-12
    )
    plt.figure(figsize=(6, 6))
    plt.scatter(img_original, cifrado_normalizado, s=1, alpha=0.3)
    plt.xlabel("Original (normalizada)")
    plt.ylabel("Cifrada (normalizada)")
    plt.title("Diagrama de dispersión: imagen original vs cifrada")
    plt.tight_layout()
    plt.savefig(str(RUTA_DISPERSION))
    print(f"[GRAFICA] Dispersión guardada en {RUTA_DISPERSION}")

def graficar_hamming(imagen, vector_cifrado, ancho, alto):
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
    cifrado_img = (cifrado_norm * 255).reshape((alto, ancho, 3)).astype(np.uint8)
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

    # Gráfica sencilla con el valor normalizado
    plt.figure(figsize=(8, 8))
    plt.bar(["Hamming"], [hamming_norm])
    plt.ylim(0, 1)
    plt.ylabel("Distancia Hamming normalizada")
    plt.title(f"Distancia Hamming (norm.): {hamming_norm:.4f}")
    plt.tight_layout()
    plt.savefig(str(RUTA_HAMMING))
    print(f"[GRAFICA] Hamming guardada en {RUTA_HAMMING}")


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
    inicio_programa = time.perf_counter()
    # 1. Cargar la imagen
    imagen, vector_inf, ancho, alto, nmax = cargar_imagen()

    # 2. Aplicar difusión
    difusion, vector_logistico, tiempo_difusion = aplicar_difusion(vector_inf, nmax)

    # 3. Aplicar confusión
    vector_cifrado, y_sinc, t, tiempo_rossler, tiempo_confusion = aplicar_confusion(difusion, vector_logistico, nmax, ROSSLER_PARAMS)

    # 4. Preparar datos para MQTT
    data = preparar_payload(vector_cifrado, y_sinc, t, ancho, alto, nmax)

    # 5. Publicar en MQTT con TLS
    t_inicio_mqtt = time.perf_counter()
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.connect(BROKER, PORT, 60)
    print("[MQTT] Conectado al broker MQTT con puerto 1883")

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
    print(f"[MQTT] Tiempo de publicación MQTT con puerto 1883: {tiempo_mqtt:.4f} segundos")
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
    graficar_hamming(imagen, vector_cifrado, ancho, alto)

    print("[PROGRAMA] Proceso de cifrado completado")

if __name__ == "__main__":
    main()
