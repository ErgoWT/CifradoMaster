import json
import time
import ssl
from pathlib import Path

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from PIL import Image

# Configuracion mqtt
BROKER = "192.168.0.55"
PORT = 8883
QOS = 1
USERNAME = "usuario1"
PASSWORD = "qwerty123"
TOPIC_KEYS = "chaosPD/keys"
TOPIC_DATA = "chaosPD/data"
CA_CERT_PATH = "/etc/mosquitto/ca_certificates/ca.crt"

# Parametros del sistema
ROSSLER_PARAMS = {
    "a": 0.2,
    "b": 0.2,
    "c": 5.7
}
H = 0.01
Y0 = [0.1, 0.1, 0.1]
KEYSTREAM = 20000

# Rutas y archivos
CARPETA_RESULTADOS = Path("Resultados_ControlPD")

def rossler_maestro(t, state, a, b, c):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

def solucion_rossler(ROSSLER_PARAMS):
    t_span = (0, H * KEYSTREAM)
    t_eval = np.arange(0, H * KEYSTREAM, H)

    sol = solve_ivp(
        fun = rossler_maestro,
        y0 = Y0,
        args = tuple(ROSSLER_PARAMS.values()),
        t_span = t_span,
        t_eval = t_eval,
        method = "RK45",
        rtol = 1e-8,
        atol = 1e-8
    )

    print("Solucion del sistema Rossler completada.")

    x = sol.y[0]
    y = sol.y[1]
    z = sol.y[2]
    t = sol.t

    return x, y, z, t

def preparar_payload(x, y, z, t):
    data = {
        "x": x.tolist(),
        "y": y.tolist(),
        "z": z.tolist(),
        "t": t.tolist()
    }
    payload = json.dumps(data)
    return payload

def main():
    x, y, z, t = solucion_rossler(ROSSLER_PARAMS)

    payload = preparar_payload(x, y, z, t)

    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set(ca_certs=CA_CERT_PATH, tls_version=ssl.PROTOCOL_TLS_CLIENT)
    client.tls_insecure_set(False)
    client.connect(BROKER, PORT, 60)
    print("Conectado al broker MQTT")

    client.publish(TOPIC_DATA, payload, qos=QOS, retain = True)
    time.sleep(0.5)
    print("Datos publicados en el topic:", TOPIC_DATA)
    client.publish(TOPIC_KEYS, json.dumps(
        {
            "a": ROSSLER_PARAMS["a"],
            "b": ROSSLER_PARAMS["b"],
            "c": ROSSLER_PARAMS["c"],
        }
    ), qos = QOS, retain = True)
    time.sleep(0.5)
    print("Parametros publicados en el topic:", TOPIC_KEYS)

    client.disconnect()

if __name__ == "__main__":
    main()