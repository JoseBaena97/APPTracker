from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Color, RoundedRectangle
import cv2
from logic import process_frame, are_both_arms_raised, is_squat_position, process_frame_fall_detection, process_frame_training, detect_fall  # Importamos la función de lógica
from logic import reset_landmark_history

# Clase para los botones con bordes redondeados
class RoundedButton(Button):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ''  # Sin imagen de fondo
        self.background_color = (0, 0, 0, 0)  # Sin fondo sólido
        self.border_radius = 15  # Radio de los bordes

        # Dibujo de los bordes redondeados
        with self.canvas.before:
            Color(0.278, 0.294, 0.302, 1)  # Color del fondo (gris oscuro)
            self.rect = RoundedRectangle(
                pos=self.pos, size=self.size, radius=[self.border_radius] * 4
            )

        # Llamamos a la actualización del rectángulo cada vez que el tamaño cambie
        self.bind(size=self.update_rect, pos=self.update_rect)

    def update_rect(self, *args):
        """Actualizar el tamaño y la posición del rectángulo redondeado"""
        self.rect.pos = self.pos
        self.rect.size = self.size

class CameraWidget(Image):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Cambiado a cv2.CAP_DSHOW para Windows
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

    def update_frame(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        # Redimensionar el frame a un tamaño fijo
        frame = cv2.resize(frame, (1920, 1080))  # Ajustar a tamaño deseado

        # Llamada a la función de procesamiento
        frame_processed, smoothed_landmarks = process_frame(frame)

        # Convertir el frame en textura
        buffer = frame_processed.tobytes()
        texture = Texture.create(size=(frame_processed.shape[1], frame_processed.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.texture = texture

    def release(self):
        self.capture.release()

class MenuScreen(BoxLayout):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.orientation = 'vertical'
        self.spacing = 10
        self.padding = 20

        self.add_widget(Label(text="Menú Principal", font_size='30sp'))

        # Botón para Modo Captura
        start_button = RoundedButton(text="Iniciar Captura", size_hint=(1, 0.2))
        start_button.bind(on_press=self.start_capture)
        self.add_widget(start_button)

        # Botón para Modo Entrenamiento
        training_button = RoundedButton(text="Modo Entrenamiento", size_hint=(1, 0.2))
        training_button.bind(on_press=self.start_training)
        self.add_widget(training_button)

        # Botón para Detección de Caídas
        fall_detection_button = RoundedButton(text="Modo Detección de Caídas", size_hint=(1, 0.2))
        fall_detection_button.bind(on_press=self.start_fall_detection)
        self.add_widget(fall_detection_button)

        # Botón de Salir
        exit_button = RoundedButton(text="Salir", size_hint=(1, 0.2))
        exit_button.bind(on_press=self.exit_app)
        self.add_widget(exit_button)

    def start_capture(self, instance):
        self.app.show_camera()

    def start_training(self, instance):
        self.app.start_training_mode()

    def start_fall_detection(self, instance):
        self.app.start_fall_detection_mode()  # Llama al método de la app principal

    def exit_app(self, instance):
        App.get_running_app().stop()


class PoseTrackerApp(App):
    def build(self):
        self.menu_screen = MenuScreen(self)
        return self.menu_screen

    def show_camera(self):
        # Crear layout principal para la cámara
        main_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        self.camera_widget = CameraWidget()
        main_layout.add_widget(self.camera_widget)

        # Layout inferior para botones y mensajes
        controls_layout = BoxLayout(size_hint_y=0.2, spacing=10)
        start_button = RoundedButton(text="Detener Captura", size_hint=(0.3, 1))
        start_button.bind(on_press=self.stop_capture)  # Botón para detener la captura
        self.message_label = Label(text="Captura en progreso...", size_hint=(0.7, 1))

        controls_layout.add_widget(start_button)
        controls_layout.add_widget(self.message_label)

        main_layout.add_widget(controls_layout)

        # Cambiar el contenido de la aplicación
        self.menu_screen.clear_widgets()
        self.menu_screen.add_widget(main_layout)

    def stop_capture(self, instance):
        self.camera_widget.release()
        self.menu_screen.clear_widgets()
        self.menu_screen.add_widget(MenuScreen(self))  # Regresar al menú principal

    def on_stop(self):
        self.camera_widget.release()

    def start_training_mode(self):
        reset_landmark_history()

        # Detener el modo de detección de caídas si está activo
        if hasattr(self, 'fall_detection_widget') and self.fall_detection_widget:
            self.fall_detection_widget.release()
            self.fall_detection_widget = None

        # Layout principal para el modo entrenamiento
        main_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        self.training_widget = TrainingWidget()  # Agregaremos esta clase
        main_layout.add_widget(self.training_widget)

        # Botón para detener el modo entrenamiento
        stop_button = RoundedButton(text="Detener Entrenamiento", size_hint=(1, 0.2))
        stop_button.bind(on_press=self.stop_training_mode)
        main_layout.add_widget(stop_button)

        # Cambiar el contenido de la aplicación
        self.menu_screen.clear_widgets()
        self.menu_screen.add_widget(main_layout)

    def stop_training_mode(self, instance=None):
            if hasattr(self, 'training_widget') and self.training_widget:
                self.training_widget.release()
                self.training_widget = None  # Liberar recursos del entrenamiento
            self.show_menu()  # Regresar al menú principal

    def on_stop(self):
        if hasattr(self, 'training_widget') and self.training_widget:
            self.training_widget.release()
        if hasattr(self, 'fall_detection_widget') and self.fall_detection_widget:
            self.fall_detection_widget.release()


    def start_fall_detection_mode(self):
        reset_landmark_history()

        # Detener el modo de entrenamiento si está activo
        if hasattr(self, 'training_widget') and self.training_widget:
            self.training_widget.release()
            self.training_widget = None

        # Crear la instancia del widget de detección de caídas
        self.fall_detection_widget = FallDetectionWidget(self)
        self.menu_screen.clear_widgets()
        self.menu_screen.add_widget(self.fall_detection_widget.main_layout)

    def stop_fall_detection_mode(self, instance=None):
        if hasattr(self, 'fall_detection_widget') and self.fall_detection_widget:
            self.fall_detection_widget.release()
            self.fall_detection_widget = None  # Liberar recursos del entrenamiento
        self.show_menu()  # Regresar al menú principal

    def show_menu(self):
        # Regresar al menú principal
        self.menu_screen.clear_widgets()
        self.menu_screen.add_widget(MenuScreen(self))

class TrainingWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'

        # Etiqueta para mostrar el mensaje de feedback
        self.feedback_label = Label(
            text="Inicio del entrenamiento",
            font_size='20sp',
            size_hint=(1, 0.1),
            halign='center',
            valign='middle'
        )
        self.feedback_label.bind(size=self.feedback_label.setter('text_size'))  # Ajustar texto automáticamente
        self.add_widget(self.feedback_label)

        # Widget para mostrar la cámara
        self.image_widget = Image()
        self.add_widget(self.image_widget)

        # Etiqueta para mostrar el progreso
        self.progress_label = Label(
            text="Progreso: 0/0",
            font_size='20sp',
            size_hint=(1, 0.1),
            halign='center',
            valign='middle'
        )
        self.progress_label.bind(size=self.progress_label.setter('text_size'))
        self.add_widget(self.progress_label)

        # Inicializar la captura de video
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Lista de objetivos y variables de estado
        self.objectives = [("Levantar brazos", 5), ("Hacer sentadillas", 5)]
        self.current_objective = 0
        self.current_count = 0
        self.feedback_message = ""

        # Variables de estado para detectar transiciones
        self.arms_were_raised = False  # Para levantamiento de brazos
        self.was_in_squat = False      # Para posición de sentadilla

        # Programar la actualización del frame
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

    def update_frame(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        try:
            # Procesar el frame y obtener landmarks
            result = process_frame_training(frame)

            # Verificar si la función devolvió resultados válidos
            if result is None:
                self.feedback_message = "No se detecta al usuario. Ajuste su posición."
                self.feedback_label.text = self.feedback_message
                return

            frame_processed, smoothed_landmarks = result

            # Verificar si los landmarks son válidos antes de continuar
            if smoothed_landmarks is None:
                self.feedback_message = "No se detecta al usuario. Ajuste su posición."
                self.feedback_label.text = self.feedback_message
                return

            # Lógica para evaluar el progreso del entrenamiento
            if self.objectives[self.current_objective][0] == "Levantar brazos":
                # Detectar si ambos brazos están levantados
                arms_raised_now = are_both_arms_raised(smoothed_landmarks)

                # Detectar transición de estado: de no levantados a levantados
                if arms_raised_now and not self.arms_were_raised:
                    self.current_count += 1  # Solo contar si hay una transición
                    self.feedback_message = f"Levantamiento detectado ({self.current_count}/{self.objectives[self.current_objective][1]})"

                # Actualizar el estado anterior
                self.arms_were_raised = arms_raised_now

            elif self.objectives[self.current_objective][0] == "Hacer sentadillas":
                # Detectar si está en posición de sentadilla
                in_squat_now = is_squat_position(smoothed_landmarks)

                # Detectar transición de estado: de no en sentadilla a en sentadilla
                if in_squat_now and not self.was_in_squat:
                    self.current_count += 1  # Solo contar si hay una transición
                    self.feedback_message = f"Sentadilla detectada ({self.current_count}/{self.objectives[self.current_objective][1]})"

                # Actualizar el estado anterior
                self.was_in_squat = in_squat_now

            # Validar si se alcanzó el objetivo actual
            if self.current_count >= self.objectives[self.current_objective][1]:
                self.current_count = 0
                self.current_objective += 1
                if self.current_objective >= len(self.objectives):
                    self.feedback_message = "¡Entrenamiento Completado!"
                    self.current_objective = 0  # Reiniciar el entrenamiento
                else:
                    self.feedback_message = f"Objetivo completado. Siguiente: {self.objectives[self.current_objective][0]}"

            # Actualizar las etiquetas
            self.feedback_label.text = self.feedback_message
            self.progress_label.text = f"Progreso: {self.current_count}/{self.objectives[self.current_objective][1]}"

            # Convertir el frame a textura para Kivy
            buffer = frame_processed.tobytes()
            texture = Texture.create(size=(frame_processed.shape[1], frame_processed.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.image_widget.texture = texture

        except Exception as e:
            print(f"Error en el modo entrenamiento: {e}")
            self.feedback_label.text = "Error al procesar el frame. Verifique la cámara."

    def release(self):
        """Libera los recursos de la cámara y detiene las actualizaciones."""
        if self.capture.isOpened():
            self.capture.release()
        Clock.unschedule(self.update_frame)


class FallDetectionWidget(Image):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Etiqueta para mostrar el estado de detección
        self.feedback_label = Label(text="Iniciando detección de caídas...", font_size='20sp', size_hint=(1, 0.1))

        # Botón para detener el modo de detección
        stop_button = RoundedButton(text="Detener Detección", size_hint=(1, 0.1))
        stop_button.bind(on_press=self.stop_fall_detection)

        # Layout principal
        self.main_layout = BoxLayout(orientation='vertical')
        self.main_layout.add_widget(self)
        self.main_layout.add_widget(self.feedback_label)
        self.main_layout.add_widget(stop_button)

        # Programar actualizaciones
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

    def update_frame(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            self.feedback_label.text = "Error de cámara. Revisa la conexión."
            return

        try:
            result = process_frame_fall_detection(frame)
            if result is None:
                self.feedback_label.text = "No se detectan landmarks. Ajuste su posición."
                return

            frame_processed, smoothed_landmarks = result

            # Detectar caídas
            if detect_fall(smoothed_landmarks):
                self.feedback_label.text = "¡Caida detectada!"
            else:
                self.feedback_label.text = "Monitoreando caídas..."

            # Actualizar el frame en la pantalla
            buffer = frame_processed.tobytes()
            texture = Texture.create(size=(frame_processed.shape[1], frame_processed.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = texture

        except Exception as e:
            print(f"Error en el modo detección de caídas: {e}")
            self.feedback_label.text = "Error al procesar el frame. Verifique la cámara."

    def stop_fall_detection(self, instance):
        """Método para detener la detección de caídas."""
        print("Deteniendo detección de caídas...")
        self.app.stop_fall_detection_mode()

    def release(self):
        """Libera los recursos de la cámara y detiene las actualizaciones."""
        if self.capture.isOpened():
            self.capture.release()
        Clock.unschedule(self.update_frame)

if __name__ == '__main__':
    PoseTrackerApp().run()
