import cv2
import mediapipe as mp
import serial
import time

# Configurar serial
try:
    ser = serial.Serial('COM4', 9600, timeout=1)
    time.sleep(2)
except serial.SerialException:
    print("Error al abrir el puerto serial.")
    exit()

# Configuración MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuración cámara
videoCap = cv2.VideoCapture(0)

PALMA_LANDMARKS = {'pulgar': 5, 'indice': 5, 'medio': 9, 'anular': 13, 'menique': 17}
PUNTA_LANDMARKS = {'pulgar': 4, 'indice': 8, 'medio': 12, 'anular': 16, 'menique': 20}

threshold = 0.03

with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                    min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    try:
        while True:
            ret, frame = videoCap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            angulos = {'pulgar': 0, 'indice': 0, 'medio': 0, 'anular': 0, 'menique': 0}
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                h, w, _ = frame.shape

                # Dibuja círculos en las puntas y bases de dedos con desplazamiento en bases
                for dedo in PUNTA_LANDMARKS:
                    punta = landmarks.landmark[PUNTA_LANDMARKS[dedo]]
                    base = landmarks.landmark[PALMA_LANDMARKS[dedo]]

                    cx_punta, cy_punta = int(punta.x * w), int(punta.y * h)
                    
                    # Para los puntos base, agregamos un desplazamiento hacia abajo
                    desplazamiento_y = int(0.02 * h)  # 2% considerado en altura
                    cx_base, cy_base = int(base.x * w), int(base.y * h) + desplazamiento_y

                    # Dibuja círculos: verde en punta, rojo en base
                    cv2.circle(frame, (cx_punta, cy_punta), 7, (0, 255, 0), -1)
                    cv2.circle(frame, (cx_base, cy_base), 7, (0, 0, 255), -1)

                    # Calcula distancia punta-base para ángulo
                    dist = ((punta.x - base.x) ** 2 + (punta.y - base.y) ** 2) ** 0.5
                    if dist <= threshold:
                        angulos[dedo] = 180
                    else:
                        angulos[dedo] = 0

            # Enviar comandos serial
            comando = ",".join(str(angulos[dedo]) for dedo in ['pulgar', 'indice', 'medio', 'anular', 'menique']) + "\n"
            ser.write(comando.encode())

            cv2.imshow("Medio", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        ser.write("0,0,0,0,0\n".encode())
        ser.close()
        videoCap.release()
        cv2.destroyAllWindows()
