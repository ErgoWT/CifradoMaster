import numpy as np
import json
import time
import pandas as pd
import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
import ssl
from scipy.integrate import solve_ivp
from PIL import Image
import sys


def logistic_map(r: float, x0: float, n: int) -> np.ndarray:
    """Generate a logistic map sequence of length n.

    Parameters
    ----------
    r : float
        Control parameter for the logistic map.
    x0 : float
        Initial value (0 < x0 < 1).
    n : int
        Number of values to generate.

    Returns
    -------
    numpy.ndarray
        Array of length n containing the logistic sequence.
    """
    x = np.empty(n, dtype=np.float64)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1.0 - x[i - 1])
    return x


def apply_difusion(vector_inf: np.ndarray, nmax: int) -> (np.ndarray, list):
    """Apply the diffusion process using the logistic map indices.

    The diffusion procedure selects the first occurrence of indices from
    the logistic sequence and fills the remainder with unused indices in
    natural order. A sentinel value (260.0) marks positions that have
    already been used in the first pass.

    Parameters
    ----------
    vector_inf : numpy.ndarray
        Flattened and normalised image vector.
    nmax : int
        Total number of elements in the image vector.

    Returns
    -------
    tuple
        A tuple containing the diffused vector and a list of indices used
        during the first pass. The list of indices is useful for
        visualisation of the permutation.
    """
    # Generate logistic map indices in the range [0, nmax)
    # For the mixture vector we normalise the logistic sequence and take floors
    # to convert to integer indices.
    # Use fixed parameters to preserve the chaotic nature (do not alter).
    r = 3.99
    x0 = 0.7
    logistic_sequence = logistic_map(r, x0, nmax)
    vector_mezcla = np.floor(logistic_sequence * nmax).astype(int)

    sentinel = 260.0
    # Make a copy to mark used indices without altering the original logistic indices
    aux = vector_mezcla.astype(float)
    difusion = np.empty_like(vector_inf)
    indices_usados = []

    k = 0  # position in the diffusion vector
    # First pass: take only the first appearance of each index
    for pos in aux:
        if pos < 0 or pos >= nmax:
            continue
        if aux[pos] != sentinel:  # check if this index has not been used
            difusion[k] = vector_inf[int(pos)]
            aux[pos] = sentinel
            indices_usados.append(int(pos))
            k += 1
        if k >= nmax:
            break

    # Second pass: fill with remaining indices in natural order
    for i in range(nmax):
        if k >= nmax:
            break
        if aux[i] != sentinel:
            difusion[k] = vector_inf[i]
            k += 1

    return difusion, indices_usados


def rossler_maestro(a: float, b: float, c: float, x0: float, y0: float, z0: float,
                    t0: float, tf: float, dt: float) -> (np.ndarray, np.ndarray):
    """Integrate the Rössler system using the given parameters.

    Parameters
    ----------
    a, b, c : float
        Parameters of the Rössler chaotic system.
    x0, y0, z0 : float
        Initial conditions for the state variables x, y, z.
    t0, tf : float
        Start and end times for the integration.
    dt : float
        Time step used to define the integration grid.

    Returns
    -------
    tuple
        Two numpy arrays: `times` containing the time points and `sol`
        containing the state values (shape len(times) x 3).
    """

    def rossler(t, state):
        x, y, z = state
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        return [dx, dy, dz]

    t_eval = np.arange(t0, tf, dt)
    sol = solve_ivp(rossler, (t0, tf), [x0, y0, z0], t_eval=t_eval, method='RK45')
    return sol.t, sol.y.T


def main():
    # Configuration variables (can be edited directly in the script)
    image_path = "Prueba.jpg"  # path to the input image
    broker = "localhost"
    port = 1883
    port_tls = 8883
    username = "user"
    password = "pass"
    topicKeys = "chaos/keys"
    topicData = "chaos/data"
    ca_cert_path = "path/to/ca.crt"  # certificate authority file path

    # Fixed chaotic parameters (do not modify)
    logisticParams = {
        "r": 3.99,
        "x0": 0.7
    }
    rosslerParams = {
        "a": 0.2,
        "b": 0.2,
        "c": 5.7,
        "x0": 0.1,
        "y0": 0.1,
        "z0": 0.1,
        "t0": 0.0,
        "tf": 50.0,
        "dt": 0.01
    }

    # Predefined synchronisation arrays (use without modification)
    time_sinc = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    keystream = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

    # 1. Load and preprocess image
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Image file not found: {image_path}")
        sys.exit(1)
    arr = np.array(img)
    alto, ancho = arr.shape[:2]
    canales = arr.shape[2] if arr.ndim == 3 else 1
    vector_inf = arr.reshape(-1) / 255.0  # normalise to [0, 1]
    nmax = vector_inf.size

    # 2. Apply diffusion
    difusion, indices_usados = apply_difusion(vector_inf, nmax)

    # 3. Apply confusion using the Rössler system
    times, traj = rossler_maestro(
        rosslerParams["a"], rosslerParams["b"], rosslerParams["c"],
        rosslerParams["x0"], rosslerParams["y0"], rosslerParams["z0"],
        rosslerParams["t0"], rosslerParams["tf"], rosslerParams["dt"]
    )
    # Extract the x-component of the trajectory as the key
    x_component = traj[:, 0]
    # Resize the key to the length of the image vector
    x_cif = np.resize(x_component, nmax)

    # Logistic sequence for confusion (reuse logistic parameters)
    logistic_seq_conf = logistic_map(logisticParams["r"], logisticParams["x0"], nmax)
    # Generate final encrypted vector
    vector_cifrado = difusion + logistic_seq_conf + x_cif

    # 4. Synchronisation: sample the y-component at positions defined by time_sinc and keystream
    y_component = traj[:, 1]
    y_sinc = []
    for t_val, k_val in zip(time_sinc, keystream):
        # Find index closest to the desired time or wrap around using keystream
        idx = int(k_val % len(times))
        y_sinc.append(float(y_component[idx]))

    # 5. Prepare JSON payloads
    keys_payload = json.dumps({
        "rosslerParams": rosslerParams,
        "logisticParams": logisticParams
    })

    data_payload = json.dumps({
        "vector_cifrado": vector_cifrado.tolist(),
        "y_sinc": y_sinc,
        "times": times.tolist(),
        "ancho": int(ancho),
        "alto": int(alto),
        "nmax": int(nmax),
        "time_sinc": time_sinc,
        "keystream": keystream
    })

    # 6. Publish keys via TLS on port 8883
    try:
        client_keys = mqtt.Client()
        client_keys.username_pw_set(username, password)
        client_keys.tls_set(ca_certs=ca_cert_path, tls_version=ssl.PROTOCOL_TLS_CLIENT)
        client_keys.tls_insecure_set(False)
        client_keys.connect(broker, port_tls, keepalive=60)
        client_keys.publish(topicKeys, keys_payload, retain=True)
        client_keys.disconnect()
    except Exception as e:
        print(f"Failed to publish keys over TLS: {e}")
        sys.exit(1)

    # 7. Publish data via non‑TLS on port 1883
    try:
        client = mqtt.Client()
        client.username_pw_set(username, password)
        client.connect(broker, port, keepalive=60)
        client.publish(topicData, data_payload, retain=True)
        client.disconnect()
    except Exception as e:
        print(f"Failed to publish data: {e}")
        sys.exit(1)

    # 8. Reconstruct images for visualisation
    # Diffusion image
    diff_img = (difusion * 255.0).reshape((alto, ancho, canales)).astype(np.uint8)
    # Confusion image (normalise the encrypted vector to [0, 255])
    cifrado_norm = (vector_cifrado - np.min(vector_cifrado))
    if np.max(cifrado_norm) != 0:
        cifrado_norm = cifrado_norm / np.max(cifrado_norm)
    else:
        cifrado_norm = cifrado_norm
    conf_img = (cifrado_norm * 255.0).reshape((alto, ancho, canales)).astype(np.uint8)

    # 9. Plot and save the combined figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Imagen Original")
    axes[0].axis('off')

    axes[1].imshow(diff_img)
    axes[1].set_title("Después de Difusión")
    axes[1].axis('off')

    axes[2].imshow(conf_img)
    axes[2].set_title("Después de Confusión")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("ImagenCifradav2.png", dpi=300)

    # 10. Plot the permutation of the first pass
    plt.figure(figsize=(8, 5))
    x_original = indices_usados
    y_nueva = list(range(len(indices_usados)))
    plt.scatter(x_original, y_nueva, s=10)
    plt.title("")  # leave blank for the user to customise
    plt.xlabel("Índice original del píxel")
    plt.ylabel("Nueva posición (primera pasada)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("grafica_primera_pasada.png", dpi=300)
    plt.show()

    # 11. Save the synchronisation trajectory to CSV
    df = pd.DataFrame({
        'times': times,
        'y_sinc': y_sinc
    })
    df.to_csv("trayectoria_y_sinc.csv", index=False)

    # End of program (one run only)
    return


if __name__ == "__main__":
    main()
