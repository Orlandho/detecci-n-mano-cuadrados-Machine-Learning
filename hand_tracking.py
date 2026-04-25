import cv2
import mediapipe as mp

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

def main():
    # Inicializar MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Iniciar captura de video
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en la ruta: {VIDEO_PATH}")
        print("Por favor, verifica que la ruta sea correcta y que el archivo exista.")
        return

    # Leer el primer frame para obtener las dimensiones
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el primer frame del video.")
        return

    height, width, _ = frame.shape

    # Definir propiedades iniciales de los cuadrados
    # Cuadrado Rojo (Esquina superior izquierda)
    red_x, red_y = 10, 10
    red_rect = [red_x, red_y, SQUARE_SIZE, SQUARE_SIZE]
    red_square_grabbed = False

    # Cuadrado Azul (Esquina superior derecha)
    blue_x = width - SQUARE_SIZE - 10
    blue_y = 10
    blue_rect = [blue_x, blue_y, SQUARE_SIZE, SQUARE_SIZE]

    # Estado del programa
    finished = False

    # Volver al inicio del video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            # Fin del video
            break

        # Convertir a RGB (MediaPipe usa RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar el frame
        results = hands.process(frame_rgb)

        # Detectar la mano
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar los landmarks de la mano (opcional pero ayuda visualmente)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calcular el centro de la palma.
                # Usaremos algunos puntos de la palma para promediar.
                cx_total, cy_total = 0, 0
                palm_landmarks = [0, 1, 2, 5, 9, 13, 17]
                for lm_id in palm_landmarks:
                    lm = hand_landmarks.landmark[lm_id]
                    cx_total += int(lm.x * width)
                    cy_total += int(lm.y * height)

                palm_cx = cx_total // len(palm_landmarks)
                palm_cy = cy_total // len(palm_landmarks)

                # Dibujar un punto en el centro de la palma calculado
                cv2.circle(frame, (palm_cx, palm_cy), 5, (0, 255, 0), cv2.FILLED)

                # Lógica de agarre
                if not red_square_grabbed:
                    # Verificar si el centro de la palma está dentro del cuadrado rojo
                    if (red_rect[0] < palm_cx < red_rect[0] + red_rect[2] and
                        red_rect[1] < palm_cy < red_rect[1] + red_rect[3]):
                        red_square_grabbed = True

                if red_square_grabbed:
                    # El cuadrado rojo se mueve con la mano (el centro de la palma)
                    red_rect[0] = palm_cx - (SQUARE_SIZE // 2)
                    red_rect[1] = palm_cy - (SQUARE_SIZE // 2)

        # Lógica de colisión
        if check_intersection(red_rect, blue_rect):
            print("El cuadrado rojo entró al azul. Terminando el programa...")
            # Mostrar mensaje en pantalla
            cv2.putText(frame, "Exito! Cuadrado rojo en azul", (width//2 - 250, height//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            finished = True

        # Dibujar los cuadrados
        # Cuadrado azul: Sin fondo (grosor = 3) en BGR -> (255, 0, 0)
        cv2.rectangle(frame, (blue_rect[0], blue_rect[1]),
                      (blue_rect[0] + blue_rect[2], blue_rect[1] + blue_rect[3]),
                      (255, 0, 0), 3)

        # Cuadrado rojo: Sin fondo (grosor = 3) en BGR -> (0, 0, 255)
        cv2.rectangle(frame, (red_rect[0], red_rect[1]),
                      (red_rect[0] + red_rect[2], red_rect[1] + red_rect[3]),
                      (0, 0, 255), 3)

        # Mostrar el frame
        cv2.imshow('Hand Tracking - Cuadrados', frame)

        # Salir si el usuario presiona 'q' o si terminamos
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if finished:
            # Mostrar el último frame con el mensaje antes de salir
            cv2.imshow('Hand Tracking - Cuadrados', frame)
            cv2.waitKey(3000)
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
