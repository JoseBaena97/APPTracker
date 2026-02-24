from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
from logic_training import process_frame_training, are_both_arms_raised, is_squat_position, reset_landmark_history

# Importaciones de KivyMD
from kivy.utils import get_color_from_hex
from kivymd.uix.fitimage import FitImage
from kivy.animation import Animation
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRectangleFlatButton
from kivymd.uix.card import MDCard
from kivymd.uix.screen import Screen
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.theming import ThemeManager
from kivymd.uix.button import MDFillRoundFlatButton

# Pantalla de menú principal con KivyMD
class MenuScreen(Screen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app

        # Barra de herramientas (toolbar)
        self.toolbar = MDTopAppBar(
            title="Training APP",
            pos_hint={"top": 1},
            elevation=10,
            md_bg_color=get_color_from_hex("#b90101"),  # Verde en hexadecimal
            specific_text_color=get_color_from_hex("#FFFFFF"),  # Blanco en hexadecimal
            left_action_items=[["menu", lambda x: self.open_navigation_drawer()]],
            right_action_items=[["exit-to-app", lambda x: self.exit_app()]],
        )
        self.add_widget(self.toolbar)

        # Layout principal
        self.layout = BoxLayout(
            orientation="vertical",
            spacing=20,
            padding=40,
            size_hint_y=None,
            height="600dp",
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )

        
        # Animación para mostrar los botones
        def animate_buttons(self):
            for button in self.card.children:
                button.opacity = 0
                Animation(opacity=1, duration=0.5).start(button)

        # Tarjeta para los botones
        self.card = MDCard(
            orientation="vertical",
            size_hint=(0.8, None),
            height="400dp",
            padding=20,
            spacing=20,
            elevation=10, #Sombra
            radius=[30], #Bordes redondeados
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )
        # Contenedor para el logo con marco redondeado
        self.logo_card = MDCard(
            size_hint=(None, None),
            size=("250dp","250dp"),
            elevation=5,
            radius=[125],
            padding=10,
            pos_hint={"center_x": 0.5, "top": 1},
        )

        # Imagen del logo (ajusta automáticamente al tamaño del contenedor)
        self.logo = FitImage(
            source="logo.png",  # Ruta de la imagen del logo
            size_hint=(1, 1),
            radius=self.logo_card.radius,  # Bordes redondeados
        )
        self.logo_card.add_widget(self.logo)

        # Agregar el logo a la tarjeta
        self.card.add_widget(self.logo_card)

        # Botón para iniciar el entrenamiento
        start_button = MDFillRoundFlatButton(
            text="Iniciar Entrenamiento",
            size_hint=(1, None),
            height="50dp",
            md_bg_color=get_color_from_hex("#b90101"),  # Verde en hexadecimal
            text_color=get_color_from_hex("#FFFFFF"),  # Blanco en hexadecimal
            on_release=self.start_training,
        )
        self.card.add_widget(start_button)

        # Botón para salir de la aplicación
        exit_button = MDFillRoundFlatButton(
            text="Salir",
            size_hint=(1, None),
            height="50dp",
            md_bg_color=get_color_from_hex("#b90101"),  # Verde en hexadecimal
            text_color=get_color_from_hex("#FFFFFF"),  # Blanco en hexadecimal
            on_release=self.exit_app,
        )
        self.card.add_widget(exit_button)

        # Agregar la tarjeta al layout
        self.layout.add_widget(self.card)
        self.add_widget(self.layout)

    def start_training(self, instance):
        self.app.start_training_mode()  # Llama al método de la aplicación principal

    def exit_app(self, instance):
        MDApp.get_running_app().stop()  # Cierra la aplicación

# Aplicación principal
class TrainingApp(MDApp):
    def build(self):
        # Configurar el tema de KivyMD
        self.theme_cls = ThemeManager()
        self.theme_cls.primary_palette = "Teal"  # Color principal
        self.theme_cls.theme_style = "Dark"  # Tema claro

        # Pantalla de menú principal
        self.menu_screen = MenuScreen(self, name="menu")
        return self.menu_screen

    def start_training_mode(self):
        # Limpia la pantalla de menú
        self.menu_screen.clear_widgets()

        # Resetear el historial de landmarks
        reset_landmark_history()

        # Layout principal para el modo entrenamiento
        self.layout = BoxLayout(orientation='vertical')
        self.feedback_label = MDLabel(
            text="Inicio del entrenamiento",
            halign="center",
            font_style="H6",
            size_hint=(1, 0.1),
        )
        self.layout.add_widget(self.feedback_label)

        # Widget para mostrar la cámara
        self.image_widget = Image()
        self.layout.add_widget(self.image_widget)

        # Botón para detener el entrenamiento y volver al menú
        stop_button = MDRectangleFlatButton(
            text="Detener Entrenamiento",
            size_hint=(1, 0.2),
            on_release=self.stop_training_mode,
        )
        self.layout.add_widget(stop_button)

        # Inicializar la captura de video
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

        # Cambiar a la pantalla de entrenamiento
        self.menu_screen.add_widget(self.layout)

    def stop_training_mode(self, instance):
        # Detener la captura de video y volver al menú principal
        if self.capture.isOpened():
            self.capture.release()
        self.menu_screen.clear_widgets()
        self.menu_screen.add_widget(MenuScreen(self, name="menu"))

    def update_frame(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        # Procesar el frame en el modo entrenamiento
        frame_processed, smoothed_landmarks = process_frame_training(frame)

        if smoothed_landmarks:
            # Verificar si ambos brazos están levantados
            arms_raised = are_both_arms_raised(smoothed_landmarks)
            if arms_raised:
                self.feedback_label.text = "Brazos levantados"

            # Verificar si está en posición de sentadilla
            squat_position = is_squat_position(smoothed_landmarks)
            if squat_position:
                self.feedback_label.text = "Sentadilla detectada"

        # Convertir el frame a textura para Kivy
        buffer = frame_processed.tobytes()
        texture = Texture.create(size=(frame_processed.shape[1], frame_processed.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image_widget.texture = texture

    def on_stop(self):
        # Liberar la cámara al cerrar la aplicación
        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()

if __name__ == '__main__':
    TrainingApp().run()