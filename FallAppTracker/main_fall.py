from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
from logic_fall import process_frame_fall_detection, reset_landmark_history
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.uix.screenmanager import Screen
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRectangleFlatButton
from kivymd.uix.card import MDCard
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.theming import ThemeManager
from kivymd.uix.button import MDFillRoundFlatButton

# Pantalla de menú principal con KivyMD
class MenuScreen(Screen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app

        # Barra de herramientas
        self.toolbar = MDTopAppBar(
            title="Modo Detección de Caídas",
            pos_hint={"top": 1},
            elevation=10,
        )
        self.add_widget(self.toolbar)

        # Layout principal
        self.layout = BoxLayout(
            orientation="vertical",
            spacing=20,
            padding=40,
            size_hint_y=None,
            height="400dp",
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )

        # Imagen de fondo
        self.background_image = Image(
            source='fondo.png',  # Aquí pones la ruta de tu imagen
            allow_stretch=True,  # Para que la imagen se estire y ocupe toda la pantalla
            keep_ratio=False,  # Desactivamos la relación de aspecto para llenar toda la pantalla
        )
        self.background_image.size = self.size  # Hacemos que ocupe todo el tamaño del layout
        self.layout.add_widget(self.background_image)

        # Tarjeta para botones
        self.card = MDCard(
            orientation="vertical",
            size_hint=(0.8, None),
            height="300dp",
            padding=20,
            spacing=20,
            elevation=10,
            radius=[30],
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )

        # Botón para iniciar
        start_button = MDFillRoundFlatButton(
            text="Iniciar Detección",
            size_hint=(1, None),
            height="50dp",
            on_release=self.start_fall_detection,
        )
        self.card.add_widget(start_button)

        # Botón para salir
        exit_button = MDFillRoundFlatButton(
            text="Salir",
            size_hint=(1, None),
            height="50dp",
            on_release=self.exit_app,
        )
        self.card.add_widget(exit_button)

        self.layout.add_widget(self.card)
        self.add_widget(self.layout)

    def start_fall_detection(self, instance):
        self.app.start_fall_detection_mode()

    def exit_app(self, instance):
        MDApp.get_running_app().stop()


class FallDetectionApp(MDApp):
    def build(self):
        self.theme_cls = ThemeManager()
        self.theme_cls.primary_palette = "Red"
        self.theme_cls.theme_style = "Light"

        self.menu_screen = MenuScreen(self, name="menu")
        return self.menu_screen
    
    def _update_feedback_bg(self, instance, value):
        self.feedback_bg_rect.size = instance.size
        self.feedback_bg_rect.pos = instance.pos

    def start_fall_detection_mode(self):
        self.menu_screen.clear_widgets()
        reset_landmark_history()

        # Layout detección
        self.layout = BoxLayout(orientation='vertical')

        # Fondo de la pantalla
        with self.layout.canvas.before:
            self.bg_color = Color(1, 1, 1, 1)  # Blanco por defecto
            self.bg_rect = Rectangle(size=self.layout.size, pos=self.layout.pos)

        # Hacer que el fondo cambie cuando el layout cambie de tamaño
        self.layout.bind(size=self._update_bg_color, pos=self._update_bg_color)

        self.feedback_container = BoxLayout(size_hint=(1, 0.1))
        with self.feedback_container.canvas.before:
            self.feedback_bg_color = Color(1, 1, 1, 1)  # Blanco por defecto
            self.feedback_bg_rect = Rectangle(size=self.feedback_container.size, pos=self.feedback_container.pos)

        self.feedback_container.bind(size=self._update_feedback_bg, pos=self._update_feedback_bg)

        self.feedback_label = MDLabel(
            text="Iniciando detección de caídas...",
            halign="center",
            font_style="H6",
            font_name="Roboto",
        )
        self.feedback_container.add_widget(self.feedback_label)
        self.layout.add_widget(self.feedback_container)
        self.image_widget = Image()
        self.layout.add_widget(self.image_widget)

        stop_button = MDFillRoundFlatButton(
            text="Detener detección",
            size_hint=(None, None),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            height="75dp",  # Mantener el tamaño del botón
            md_bg_color=(1, 0, 0, 1),  # Fondo rojo (rojo claro) para el botón
            text_color=(0, 0, 0, 1),  # Texto blanco
            font_name="Roboto-Bold",
            font_size="25sp",  # Hacer el texto más grande
            on_release=self.stop_fall_detection_mode,
        )
        self.layout.add_widget(stop_button)

        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

        self.menu_screen.add_widget(self.layout)

    def stop_fall_detection_mode(self, instance):
        if self.capture.isOpened():
            self.capture.release()
        self.menu_screen.clear_widgets()
        self.menu_screen.add_widget(MenuScreen(self, name="menu"))

    def update_frame(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        # 👇 Este es el frame procesado y el flag de caída
        frame_processed, fall_detected = process_frame_fall_detection(frame)

        if fall_detected:
            self.feedback_label.text = "¡Caída detectada!"
            self.feedback_bg_color.rgba = (1, 0, 0, 1)  # Rojo en el área del mensaje
            self.bg_color.rgba = (1, 0, 0, 1)  # Rojo en todo el fondo de la pantalla
            self.feedback_label.font_size = '48sp'
        else:
            self.feedback_label.text = "Monitoreando caídas..."
            self.feedback_bg_color.rgba = (1, 1, 1, 1)  # Blanco en el área del mensaje
            self.bg_color.rgba = (1, 1, 1, 1)  # Blanco en todo el fondo de la pantalla
            self.feedback_label.font_size = '48sp'

        # Mostrar la imagen en pantalla
        buffer = frame_processed.tobytes()
        texture = Texture.create(size=(frame_processed.shape[1], frame_processed.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image_widget.texture = texture

    def _update_feedback_bg(self, instance, value):
        self.feedback_bg_rect.size = instance.size
        self.feedback_bg_rect.pos = instance.pos

    def _update_bg_color(self, instance, value):
        self.bg_rect.size = instance.size
        self.bg_rect.pos = instance.pos

    def on_stop(self):
        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()


if __name__ == '__main__':
    FallDetectionApp().run()
