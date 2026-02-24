import cv2
import mediapipe as mp
import collections
import math

# Inicializa MediaPipe para la detección de poses
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=False)
drawing_utils = mp.solutions.drawing_utils

# Tamaño del filtro de media móvil para el suavizado
SMOOTHING_WINDOW_SIZE = 5

# Inicializar landmark_history como una variable global
landmark_history = collections.defaultdict(lambda: collections.deque(maxlen=SMOOTHING_WINDOW_SIZE))

def reset_landmark_history():
    """Resetea el historial de landmarks cuando se cambia de modo."""
    global landmark_history
    landmark_history.clear()  # Limpiar el historial existente

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
        if landmark_history[i]:  # Ahora landmark_history está definida
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

# Función para detectar caídas basándose en la posición de la cadera y la cabeza
def detect_fall(smoothed_landmarks):
    if not isinstance(smoothed_landmarks, list) or len(smoothed_landmarks) == 0:
        return False  # Si no hay landmarks, no hay detección

    # Obtener las coordenadas de la cadera y la nariz (o cabeza)
    hip_y = smoothed_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1]
    nose_y = smoothed_landmarks[mp_pose.PoseLandmark.NOSE.value][1]

    # Verificar si la cabeza está a la misma altura o por debajo de la cadera (indicativo de caída)
    if nose_y >= hip_y:
        return True

    return False

# Función para procesar el frame en el modo de detección de caídas
def process_frame_fall_detection(frame):
    # Convertir a RGB para MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    h, w, _ = frame.shape

    smoothed_landmarks = None
    fall_detected = False  # Inicializar fall_detected con un valor predeterminado

    if results.pose_landmarks:
        smoothed_landmarks = smooth_landmarks(results.pose_landmarks.landmark)

        # Dibujar los elementos en el frame
        draw_body_connections(frame, smoothed_landmarks, h, w)
        draw_wrist_points(frame, smoothed_landmarks, h, w)
        draw_torso_line(frame, smoothed_landmarks, h, w)
        draw_head_and_neck_tracking(frame, smoothed_landmarks, h, w)
        draw_joint_points(frame, smoothed_landmarks, h, w)

         # Detectar si ocurrió una caída
        if smoothed_landmarks:  # Verificar que smoothed_landmarks no sea None
            fall_detected = detect_fall(smoothed_landmarks)
            

    # Voltear la imagen verticalmente (de arriba a abajo)
    frame = cv2.flip(frame, 0)

    return frame, fall_detected