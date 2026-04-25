import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
# Cambia esta variable con la ruta de tu video MP4.
VIDEO_PATH = 'video.mp4'

# Tamaño de los cuadrados
SQUARE_SIZE = 100

def check_intersection(rect1, rect2):
    """
    Verifica si dos rectángulos se intersectan.
    rect = (x, y, w, h)
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    if (x1 < x2 + w2 and
        x1 + w1 > x2 and
        y1 < y2 + h2 and
        y1 + h1 > y2):
        return True
    return False

class HandTrackingApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Inicializar MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Iniciar captura de video
        self.cap = cv2.VideoCapture(VIDEO_PATH)

        if not self.cap.isOpened():
            print(f"Error: No se pudo abrir el video en la ruta: {VIDEO_PATH}")
            messagebox.showerror("Error", f"No se pudo abrir el video: {VIDEO_PATH}")
            self.window.destroy()
            return

        # Leer el primer frame para obtener las dimensiones
        ret, frame = self.cap.read()
        if not ret:
            print("Error: No se pudo leer el primer frame del video.")
            messagebox.showerror("Error", "No se pudo leer el video.")
            self.window.destroy()
            return

        self.height, self.width, _ = frame.shape
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Volver al inicio

        # Configurar la ventana de Tkinter con el tamaño del video
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        # Botón para salir
        self.btn_quit = tk.Button(window, text="Salir", width=10, command=self.close_app)
        self.btn_quit.pack(pady=5)

        # Variables de estado
        self.red_x, self.red_y = 10, 10
        self.red_rect = [self.red_x, self.red_y, SQUARE_SIZE, SQUARE_SIZE]
        self.red_square_grabbed = False

        self.blue_x = self.width - SQUARE_SIZE - 10
        self.blue_y = 10
        self.blue_rect = [self.blue_x, self.blue_y, SQUARE_SIZE, SQUARE_SIZE]

        self.finished = False

        # Iniciar el bucle de actualización
        self.delay = 15 # milisegundos
        self.update_frame()

    def update_frame(self):
        if self.finished:
            return

        ret, frame = self.cap.read()
        if not ret:
            # Fin del video
            self.close_app()
            return

        # Procesamiento de MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                cx_total, cy_total = 0, 0
                palm_landmarks = [0, 1, 2, 5, 9, 13, 17]
                for lm_id in palm_landmarks:
                    lm = hand_landmarks.landmark[lm_id]
                    cx_total += int(lm.x * self.width)
                    cy_total += int(lm.y * self.height)

                palm_cx = cx_total // len(palm_landmarks)
                palm_cy = cy_total // len(palm_landmarks)

                cv2.circle(frame, (palm_cx, palm_cy), 5, (0, 255, 0), cv2.FILLED)

                if not self.red_square_grabbed:
                    if (self.red_rect[0] < palm_cx < self.red_rect[0] + self.red_rect[2] and
                        self.red_rect[1] < palm_cy < self.red_rect[1] + self.red_rect[3]):
                        self.red_square_grabbed = True

                if self.red_square_grabbed:
                    self.red_rect[0] = palm_cx - (SQUARE_SIZE // 2)
                    self.red_rect[1] = palm_cy - (SQUARE_SIZE // 2)

        if check_intersection(self.red_rect, self.blue_rect):
            print("El cuadrado rojo entró al azul. Terminando el programa...")
            cv2.putText(frame, "Exito! Cuadrado rojo en azul", (self.width//2 - 250, self.height//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            self.finished = True

        cv2.rectangle(frame, (self.blue_rect[0], self.blue_rect[1]),
                      (self.blue_rect[0] + self.blue_rect[2], self.blue_rect[1] + self.blue_rect[3]),
                      (255, 0, 0), 3)

        cv2.rectangle(frame, (self.red_rect[0], self.red_rect[1]),
                      (self.red_rect[0] + self.red_rect[2], self.red_rect[1] + self.red_rect[3]),
                      (0, 0, 255), 3)

        # Convertir frame de BGR (OpenCV) a RGB y luego a Image para Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))

        # Mostrar en el Canvas
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        if self.finished:
            # Esperar un poco antes de cerrar para que el usuario vea el mensaje
            self.window.after(3000, self.close_app)
        else:
            # Repetir el loop
            self.window.after(self.delay, self.update_frame)

    def close_app(self):
        if self.cap.isOpened():
            self.cap.release()
        self.hands.close()
        self.window.destroy()

def main():
    root = tk.Tk()
    app = HandTrackingApp(root, "Hand Tracking con Tkinter")
    root.mainloop()

if __name__ == "__main__":
    main()
