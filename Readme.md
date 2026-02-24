# 🛡️ APPTracker: Pose Tracking & Fall Detection

**APPTracker** es una aplicación de **Visión Artificial** desarrollada en Python diseñada para la monitorización de la actividad física y la seguridad personal. Utiliza la cámara del dispositivo para rastrear puntos clave del cuerpo humano (Pose Estimation), permitiendo guiar entrenamientos y detectar situaciones de emergencia como caídas accidentales.

---

## 🚀 Funcionalidades Principales

La aplicación se divide en tres modos operativos principales:

* **🎥 Modo Captura:** Visualización en tiempo real del esqueleto digital generado por MediaPipe, mostrando conexiones corporales y puntos de articulación.
* **🏋️ Modo Entrenamiento:** Un asistente interactivo que cuenta repeticiones de:
    * **Levantamiento de brazos:** Detecta cuando las muñecas superan la altura de los hombros.
    * **Sentadillas:** Valida que los pies estén a la anchura de los hombros y que las rodillas alcancen un ángulo de flexión cercano a los 90° ($75^\circ$ a $105^\circ$).
* **🚨 Detección de Caídas:** Sistema de vigilancia que dispara una alerta visual y sonora si la posición de la cabeza (nariz) desciende por debajo del nivel de la cadera.

---

## 🧠 Lógica y Algoritmos

El proyecto destaca por el tratamiento de datos biométricos en tiempo real:

### Suavizado de Movimiento (Smoothing)
Para evitar el ruido en la imagen, se implementa un filtro de media móvil. Si el movimiento es lento, se aplica un promedio ponderado ($70\%$ posición anterior, $30\%$ posición actual) para estabilizar los puntos clave.

### Detección de Sentadillas
Se utiliza trigonometría para calcular el ángulo en la rodilla ($B$) a partir de los puntos de la cadera ($A$) y el tobillo ($C$):

$$\cos(B) = \frac{\vec{BA} \cdot \vec{BC}}{|\vec{BA}| |\vec{BC}|}$$

El sistema confirma la repetición solo si el ángulo resultante está dentro del rango óptimo de entrenamiento.

---

## 🛠️ Stack Tecnológico

* **Lenguaje:** Python
* **Interfaz Gráfica:** [Kivy](https://kivy.org/) (con diseño de botones redondeados personalizados)
* **Visión Artificial:** [MediaPipe Pose](https://google.github.io/mediapipe/) & [OpenCV](https://opencv.org/)
* **Motor de Audio:** [Pygame Mixer](https://www.pygame.org/) para feedback auditivo

---

## 📦 Instalación y Uso

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/JoseBaena97/APPTracker.git](https://github.com/JoseBaena97/APPTracker.git)
    cd APPTracker
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install kivy opencv-python mediapipe pygame
    ```

3.  **Preparar recursos:**
    Asegúrate de tener un archivo llamado `feedback_sound.wav` en la raíz del proyecto para que las alertas sonoras funcionen correctamente.

4.  **Ejecutar:**
    ```bash
    python main.py
    ```

---

## 📁 Estructura del Proyecto

* `main.py`: Gestiona la interfaz de usuario, el ciclo de vida de la cámara y los cambios de pantalla en Kivy.
* `logic.py`: Contiene el motor de procesamiento de imágenes, cálculos matemáticos de ángulos, filtros de suavizado y lógica de detección de caídas.

---

## 👤 Autor
**José Baena** - [@JoseBaena97](https://github.com/JoseBaena97)
**Alicia López** - [@Alicia2202](https://github.com/Alicia2202)