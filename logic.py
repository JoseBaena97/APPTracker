import cv2
import mediapipe as mp
import collections
import math
import pygame


# Inicializa MediaPipe para la detección de poses
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=False)
drawing_utils = mp.solutions.drawing_utils

# Configuración del sonido
pygame.mixer.init()
sound = pygame.mixer.Sound("feedback_sound.wav") # Se ha de tener el .wav en la carpeta

# Tamaño del filtro de media móvil para el suavizado
SMOOTHING_WINDOW_SIZE = 5

def reset_landmark_history():
    """Resetea el historial de landmarks cuando se cambia de modo."""
    global landmark_history
    landmark_history = collections.defaultdict(lambda: collections.deque(maxlen=SMOOTHING_WINDOW_SIZE))

# Función para calcular la distancia entre dos puntos de referencia (landmarks)
def calculate_distance(lm1, lm2):
    return math.sqrt((lm1[0] - lm2[0]) ** 2 + (lm1[1] - lm2[1]) ** 2)

# Función para suavizar los landmarks
def smooth_landmarks(landmarks):
    if not landmarks:
        return []
    smoothed_landmarks = []
    for i, landmark in enumerate(landmarks):
        current_position = (landmark.x, landmark.y)
        if landmark_history[i]:
            last_position = landmark_history[i][-1]
            movement_speed = calculate_distance(last_position, current_position)
            # Usar un promedio ponderado si el movimiento es lento
            if movement_speed > 0.05:
                smoothed_landmarks.append(current_position)
            else:
                smoothed_x = (0.7 * last_position[0]) + (0.3 * current_position[0])
                smoothed_y = (0.7 * last_position[1]) + (0.3 * current_position[1])
                smoothed_landmarks.append((smoothed_x, smoothed_y))
        else:
            smoothed_landmarks.append(current_position)
        landmark_history[i].append(current_position)

    return smoothed_landmarks

# Función para dibujar las conexiones del cuerpo
def draw_body_connections(frame, smoothed_landmarks, h, w):
    body_connections = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value),
        (mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value),
        (mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value),
        (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
        (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
        (mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value),
        (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
        (mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value)
    ]
    for connection in body_connections:
        x1, y1 = int(smoothed_landmarks[connection[0]][0] * w), int(smoothed_landmarks[connection[0]][1] * h)
        x2, y2 = int(smoothed_landmarks[connection[1]][0] * w), int(smoothed_landmarks[connection[1]][1] * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Función para dibujar puntos en las muñecas
def draw_wrist_points(frame, smoothed_landmarks, h, w):
    x_lw, y_lw = int(smoothed_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][0] * w), int(smoothed_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][1] * h)
    cv2.circle(frame, (x_lw, y_lw), 6, (0, 255, 0), cv2.FILLED)

    x_rw, y_rw = int(smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][0] * w), int(smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][1] * h)
    cv2.circle(frame, (x_rw, y_rw), 6, (0, 255, 0), cv2.FILLED)

# Función para dibujar la línea vertical en el torso
def draw_torso_line(frame, smoothed_landmarks, h, w):
    mid_shoulder_x = int((smoothed_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0] + smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0]) / 2 * w)
    mid_shoulder_y = int((smoothed_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1] + smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1]) / 2 * h)
    mid_hip_x = int((smoothed_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0] + smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][0]) / 2 * w)
    mid_hip_y = int((smoothed_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1] + smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1]) / 2 * h)

    cv2.line(frame, (mid_shoulder_x, mid_shoulder_y), (mid_hip_x, mid_hip_y), (0, 255, 0), 2)

# Función para rastrear la cabeza y el cuello
def draw_head_and_neck_tracking(frame, smoothed_landmarks, h, w):
    neck_x = int((smoothed_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0] + smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0]) / 2 * w)
    neck_y = int((smoothed_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1] + smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1]) / 2 * h)

    cv2.circle(frame, (neck_x, neck_y), 6, (0, 255, 0), cv2.FILLED)

    nose_x = int(smoothed_landmarks[mp_pose.PoseLandmark.NOSE.value][0] * w)
    nose_y = int(smoothed_landmarks[mp_pose.PoseLandmark.NOSE.value][1] * h)
    cv2.circle(frame, (nose_x, nose_y), 6, (0, 255, 0), cv2.FILLED)

    cv2.line(frame, (neck_x, neck_y), (nose_x, nose_y), (0, 255, 0), 2)

# Función para dibujar puntos clave en las articulaciones
def draw_joint_points(frame, smoothed_landmarks, h, w):
    joint_indexes = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value
    ]
    for index in joint_indexes:
        x = int(smoothed_landmarks[index][0] * w)
        y = int(smoothed_landmarks[index][1] * h)
        cv2.circle(frame, (x, y), 6, (0, 255, 0), cv2.FILLED)

# Función para verificar si ambos brazos están levantados
def are_both_arms_raised(smoothed_landmarks):
    # Coordenadas de los hombros y muñecas
    left_shoulder_y = smoothed_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER][1]
    right_shoulder_y = smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER][1]
    left_wrist_y = smoothed_landmarks[mp_pose.PoseLandmark.LEFT_WRIST][1]
    right_wrist_y = smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST][1]

    # Verifica si ambos brazos están levantados
    if left_shoulder_y > left_wrist_y and right_shoulder_y > right_wrist_y:
        return True
    return False

# Función para verificar si los pies están a la altura de los hombros
def are_feet_at_shoulder_height(smoothed_landmarks):
    # Distancia entre hombros
    left_shoulder = smoothed_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_distance = calculate_distance(left_shoulder, right_shoulder)

    # Distancia entre los tobillos
    left_ankle = smoothed_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    ankle_distance = calculate_distance(left_ankle, right_ankle)

    # Verificar si la distancia entre los tobillos es similar a la distancia entre los hombros
    return abs(shoulder_distance - ankle_distance) < 0.2  # Umbral para tolerar pequeñas diferencias

# Función para verificar si la flexión de las rodillas es aproximadamente 90 grados
def is_squat_position(smoothed_landmarks):
    # Coordenadas de las caderas, rodillas y tobillos
    left_hip = smoothed_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_knee = smoothed_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    left_ankle = smoothed_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    # Calcular ángulos entre cadera, rodilla y tobillo usando los puntos de las rodillas
    def calculate_angle(a, b, c):
        # A, B y C son tres puntos, calculamos el ángulo en B
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        cos_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (math.sqrt(ba[0]**2 + ba[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2))
        angle = math.acos(cos_angle)
        return math.degrees(angle)

    # Verificar el ángulo en la rodilla (cadera -> rodilla -> tobillo)
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    # Verificar si el ángulo es cercano a 90 grados para ambas rodillas
    return 75 < left_knee_angle < 105 and 75 < right_knee_angle < 105

# Función para procesar el frame y detectar la sentadilla
def process_frame(frame):
    # Convertir a RGB para MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    h, w, _ = frame.shape

    smoothed_landmarks = None  # Inicializamos los landmarks como None

    if results.pose_landmarks:
        smoothed_landmarks = smooth_landmarks(results.pose_landmarks.landmark)

        # Verificar si ambos brazos están levantados
        if smoothed_landmarks:  # Validamos que no sea None
            arms_raised = are_both_arms_raised(smoothed_landmarks)

            # Lógica para reproducir sonido si ambos brazos están levantados
            if arms_raised:
                if not pygame.mixer.get_busy():  # Solo reproduce si no hay sonido en curso
                    sound.play()

            # Detectar si está en posición de sentadilla
            feet_at_shoulder_height = are_feet_at_shoulder_height(smoothed_landmarks)
            squat_position = is_squat_position(smoothed_landmarks)

            # Si está en posición de sentadilla, reproducir sonido y mostrar mensaje
            if feet_at_shoulder_height and squat_position:
                if not pygame.mixer.get_busy():  # Solo reproduce si no hay sonido en curso
                    sound.play()
                cv2.putText(frame, "Sentadilla detectada", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Dibujar los elementos en el frame
            draw_body_connections(frame, smoothed_landmarks, h, w)
            draw_wrist_points(frame, smoothed_landmarks, h, w)
            draw_torso_line(frame, smoothed_landmarks, h, w)
            draw_head_and_neck_tracking(frame, smoothed_landmarks, h, w)
            draw_joint_points(frame, smoothed_landmarks, h, w)

            # Mostrar mensaje si ambos brazos están levantados
            if arms_raised:
                cv2.putText(frame, "Brazos levantados", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Voltear la imagen verticalmente (de arriba a abajo)
    frame = cv2.flip(frame, 0)

    # Retornar el frame procesado y los landmarks suavizados
    return frame, smoothed_landmarks

def process_frame_training(frame):
    # Convertir a RGB para MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    h, w, _ = frame.shape

    smoothed_landmarks = None  # Inicializamos los landmarks como None

    if results.pose_landmarks:
        smoothed_landmarks = smooth_landmarks(results.pose_landmarks.landmark)

        # Verificar si ambos brazos están levantados
        if smoothed_landmarks:  # Validamos que no sea None
            arms_raised = are_both_arms_raised(smoothed_landmarks)

            # Lógica para reproducir sonido si ambos brazos están levantados
            if arms_raised:
                if not pygame.mixer.get_busy():  # Solo reproduce si no hay sonido en curso
                    sound.play()

            # Detectar si está en posición de sentadilla
            feet_at_shoulder_height = are_feet_at_shoulder_height(smoothed_landmarks)
            squat_position = is_squat_position(smoothed_landmarks)

            # Si está en posición de sentadilla, reproducir sonido y mostrar mensaje
            if feet_at_shoulder_height and squat_position:
                if not pygame.mixer.get_busy():  # Solo reproduce si no hay sonido en curso
                    sound.play()
                cv2.putText(frame, "Sentadilla detectada", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Dibujar los elementos en el frame
            draw_body_connections(frame, smoothed_landmarks, h, w)
            draw_wrist_points(frame, smoothed_landmarks, h, w)
            draw_torso_line(frame, smoothed_landmarks, h, w)
            draw_head_and_neck_tracking(frame, smoothed_landmarks, h, w)
            draw_joint_points(frame, smoothed_landmarks, h, w)

            # Mostrar mensaje si ambos brazos están levantados
            if arms_raised:
                cv2.putText(frame, "Brazos levantados", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Voltear la imagen verticalmente (de arriba a abajo)
    frame = cv2.flip(frame, 0)

    # Retornar el frame procesado y los landmarks suavizados
    return frame, smoothed_landmarks

def process_frame_fall_detection(frame):
    # Convertir a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    h, w, _ = frame.shape

    smoothed_landmarks = None

    if results.pose_landmarks:
        smoothed_landmarks = smooth_landmarks(results.pose_landmarks.landmark)

        # Dibujar los elementos en el frame
        draw_body_connections(frame, smoothed_landmarks, h, w)
        draw_wrist_points(frame, smoothed_landmarks, h, w)
        draw_torso_line(frame, smoothed_landmarks, h, w)
        draw_head_and_neck_tracking(frame, smoothed_landmarks, h, w)
        draw_joint_points(frame, smoothed_landmarks, h, w)

        # Detectar si ocurrió una caída
        fall_detected = detect_fall(smoothed_landmarks)
        if fall_detected:
                cv2.putText(frame, "Caida detectada", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Voltear la imagen verticalmente (de arriba a abajo)
        frame = cv2.flip(frame, 0)

    return frame, fall_detected


# Función para detectar caídas basándose en la posición de la cadera y la cabeza
def detect_fall(smoothed_landmarks):

    if not isinstance(smoothed_landmarks,list) or len(smoothed_landmarks) == 0:
        return False  # Si no hay landmarks, no hay detección

    # Obtener las coordenadas de la cadera y la nariz (o cabeza)
    hip_y = smoothed_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1]
    nose_y = smoothed_landmarks[mp_pose.PoseLandmark.NOSE.value][1]

    # Verificar si la cabeza está a la misma altura o por debajo de la cadera (indicativo de caída)
    if nose_y >= hip_y:
        return True

    return False

