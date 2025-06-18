import numpy as np
import json
import time
import pandas as pd
import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from PIL import Image

# ========== CONFIGURACION MQTT ==========
broker = "192.168.0.55"
port = 1883
username = "usuario1"
password = "qwerty123"
topicKeys = "chaos/keys"
topicData = "chaos/data"
# ========================================

# ========== PARAMETROS DE CIFRADO ==========
# Parámetros para Rössler
rosslerParams = {
    "a": 0.2,
    "b": 0.2,
    "c": 5.7,
}
# Parámetros para Logistic Map
logisticParams = {
    "aLog": 3.99,
    "x0_log": 0.4
}

# ========== FUNCIÓN DE DIFUSIÓN ==========
def aplicar_difusion(imagen, a_log, x0_log):
    # Se aplica la etapa de difusión utilizando
    # el mapa logistico
    # Se convierte la imagen a un vector y se normaliza
    img = np.array(imagen) # Convertir imagen a array numpy
    alto, ancho, canales = img.shape # Obtener dimensiones de la imagen
    vector_inf = img.flatten() # Convertir imagen a vector 1D
    nmax = vector_inf.size # Tamaño del vector
    vector_inf = vector_inf / 255.0 # Normalizar valores entre 0 y 1

    print("APLICANDO DIFUSIÓN...")

    # 1. Generamos el vector logístico
    vector_logistico = np.zeros(nmax)
    x = x0_log
    for i in range(nmax):
        x = a_log * x * (1 - x)
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


    print("DIFUSIÓN COMPLETADA")

    return difusion, vector_logistico, ancho, alto, nmax

def rossler_maestro(t, state, a, b, c):
    # Ecuaciones del sistema de Rössler
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

def aplicar_confusion(difusion, vector_logistico, nmax, rosslerParams):
    # Se aplica la etapa de confusión utilizando el oscilador de Rössler
    time_sinc = 2500
    keystream = 20000
    h = 0.01
    y0 = [0.1, 0.1, 0.1]  # Condiciones iniciales del sistema de Rössler
    
    print("APLICANDO CONFUSIÓN...")

    # 1. Se calculan las iteraciones totales (sincronización + cifrado)
    iteraciones = time_sinc + keystream
    print(f"Iteraciones totales: {iteraciones}")

    # 2. Resolver el sistema de Rössler
    t_span = (0, iteraciones * h)
    t_eval = np.linspace(0, iteraciones * h, iteraciones)

    solucion = solve_ivp(
        fun = rossler_maestro,
        y0 = y0,
        args = tuple(rosslerParams.values()),
        t_span = t_span,
        t_eval = t_eval,
        method = 'RK45',
        rtol = 1e-8,
        atol = 1e-8
    )

    # 3. Extraer las trayectorias del sistema de Rössler
    x = solucion.y[0][time_sinc:] # Para la confusion
    y = solucion.y[1] # Para sincronizacion
    t = solucion.t

    x_cif = np.resize(x, nmax)  # Redimensionar para que coincida con nmax

    # 4. Aplicar confusión (solo después del tiempo de sincronización)
    vector_cifrado = np.zeros(nmax)
    print("Aplicando confusión a los datos...")
    vector_cifrado = difusion + vector_logistico + x_cif
    print("Confusión aplicada correctamente")

    return vector_cifrado, y, t

# ========== PRINCIPAL ==========
if __name__ == "__main__":
    # 1. Cargar la imagen
    imagen = Image.open("Pixel_3.jpg")
    print("Imagen cargada correctamente")

    # 2. Aplicar la etapa de difusión
    difusion, vector_logistico, ancho, alto, nmax = aplicar_difusion(
        imagen,
        logisticParams['aLog'],
        logisticParams['x0_log']
    )

    print("\nPAUSA")
    time.sleep(2)
    # 3. Aplicar la etapa de confusión
    vector_cifrado, y_sinc, t = aplicar_confusion(
        difusion,
        vector_logistico,
        nmax,
        rosslerParams
    )

    # 4. Preparar datos para MQTT
    data = {
        'vector_cifrado': vector_cifrado.tolist(),
        'y_sinc': y_sinc.tolist(),
        'times': t.tolist(),
        'ancho': ancho,
        'alto': alto,
        'nmax': nmax,
        'time_sinc': 2500,
        'keystream': 20000
    }

    # 5. Configurar y enviar los datos a través de MQTT
    client = mqtt.Client()
    client.username_pw_set(username, password)

    try:
        client.connect(broker, port, 60)
        print("Conectado al broker MQTT")
        
        # Enviar parámetros por separado
        client.publish(topicKeys, json.dumps({
            'rosslerParams': rosslerParams,
            'logisticParams': logisticParams
        }), retain=True)

        # Enviar datos cifrados
        client.publish(topicData, json.dumps(data), retain=True)
        print("Datos cifrados enviados correctamente")
    except Exception as e:
        print(f"Error al enviar datos: {e}")
    finally:
        client.disconnect()
        print("Desconectado del broker MQTT")

     # 6. Visualización final
    plt.figure(figsize=(15, 5))
    
    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(imagen)
    plt.title("Original")
    plt.axis('off')
    
    # Después de difusión
    plt.subplot(1, 3, 2)
    difusion_img = (difusion * 255).reshape((alto, ancho, 3)).astype(np.uint8)
    plt.imshow(difusion_img)
    plt.title("Después de Difusión")
    plt.axis('off')
    
    # Después de confusión (pseudo-imagen)
    plt.subplot(1, 3, 3)
    cifrado_img = ((vector_cifrado - np.min(vector_cifrado)) / 
                  (np.max(vector_cifrado) - np.min(vector_cifrado)) * 255
    ).reshape((alto, ancho, 3)).astype(np.uint8)
    plt.imshow(cifrado_img)
    plt.title("Después de Confusión")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("ImagenCifradav2.png")
    print("Resultados del proceso completo guardados en ImagenCifradav2.png")
    
    # 7. Guardar componentes para análisis
    trayectoria_y = {
        't': t.tolist(),
        'y_sinc': y_sinc.tolist()
    }
    pd.DataFrame(trayectoria_y).to_csv('trayectoria_y_sinc.csv', index=False)
    
    print("\n--- CIFRADO COMPLETO ---")
