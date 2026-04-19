import cv2
import numpy as np
import time
from collections import deque

# ---------- Configuración ----------
CAMERA_ID = 0
DWELL_FRAMES = 15        # Cuántos frames mirar la misma celda para "seleccionar"
GRID_COLS = 6
GRID_ROWS = 5
CELL_W = 180             # Aumentado tamaño de celda (px)
CELL_H = 120             # Aumentado tamaño de celda (px)
MARGIN = 15
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.5         # Texto más grande

# Letras y caracteres especiales
letters = [
    'A', 'B', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'J', 'K', 'L', 
    'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X',
    'Y', 'Z', '.', ',', '!', '?'
]

# Asegurar que tengamos suficientes caracteres
while len(letters) < GRID_COLS * GRID_ROWS:
    letters.append(' ')

# ---------- Carga cascades ----------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# ---------- Estado del teclado ----------
selected_row = 0
selected_col = 0
output_text = ""
dwell_counter = 0
last_cell = (-1, -1)

# Para estabilizar la posición del pupil (media móvil)
smooth_queue_x = deque(maxlen=8)
smooth_queue_y = deque(maxlen=8)

# ---------- Sistema de Calibración ----------
calibrating = False
calibration_step = 0
calibration_data = {"gaze_points": [], "screen_points": []}
calibration_matrix = None
calibration_start_time = 0
CALIBRATION_DURATION = 3  # segundos por punto

# Puntos de calibración (coordenadas normalizadas)
calibration_targets = [
    (0.2, 0.2), (0.8, 0.2),  # Superior izquierda, Superior derecha
    (0.5, 0.5),              # Centro
    (0.2, 0.8), (0.8, 0.8)   # Inferior izquierda, Inferior derecha
]

def detect_pupil(eye_roi_gray):
    """
    Encuentra la posición aproximada de la pupila en una ROI de ojo en escala de grises.
    Retorna (cx_normalizado, cy_normalizado) o None si no encuentra.
    """
    # filtro y umbral para resaltar la pupila (zona más oscura)
    blur = cv2.medianBlur(eye_roi_gray, 5)
    _, th = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)  # invertir: pupila negra -> blanca
    # morfología para limpiar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    # buscar contornos grandes (pupila)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # escoger contorno con mayor área
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    h, w = eye_roi_gray.shape
    if area < (w*h)*0.001:  # muy pequeño -> no válido
        return None

    M = cv2.moments(c)
    if M['m00'] == 0:
        return None
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # normalizar
    nx = cx / float(w)
    ny = cy / float(h)
    return nx, ny, th, (cx, cy)

def start_calibration():
    """Inicia el proceso de calibración"""
    global calibrating, calibration_step, calibration_data, calibration_start_time
    calibrating = True
    calibration_step = 0
    calibration_data = {"gaze_points": [], "screen_points": []}
    calibration_start_time = time.time()
    print("Iniciando calibración... Mira los puntos que aparecen en pantalla.")

def update_calibration():
    """Actualiza la calibración con los datos recolectados"""
    global calibration_matrix
    
    if len(calibration_data["gaze_points"]) < 5:
        print("No hay suficientes datos para calibrar")
        return
    
    # Convertir a arrays de numpy
    gaze_points = np.array(calibration_data["gaze_points"], dtype=np.float32)
    screen_points = np.array(calibration_data["screen_points"], dtype=np.float32)
    
    # Calcular matriz de transformación afín
    try:
        # Usar transformación afín (2x3) para mapear coordenadas de mirada a pantalla
        calibration_matrix, _ = cv2.estimateAffine2D(gaze_points, screen_points, method=cv2.RANSAC)
        if calibration_matrix is not None:
            print("Calibración completada exitosamente!")
            print(f"Matriz de calibración:\n{calibration_matrix}")
        else:
            print("Error en la calibración: matriz no calculada")
    except Exception as e:
        print(f"Error en calibración: {e}")

def apply_calibration(gaze_x, gaze_y):
    """Aplica la matriz de calibración a las coordenadas de mirada"""
    if calibration_matrix is None:
        return gaze_x, gaze_y
    
    # Crear punto de entrada
    point = np.array([gaze_x, gaze_y, 1], dtype=np.float32)
    
    # Aplicar transformación
    calibrated_point = calibration_matrix @ point
    
    # Asegurar que esté en rango [0,1]
    calibrated_x = np.clip(calibrated_point[0], 0.0, 1.0)
    calibrated_y = np.clip(calibrated_point[1], 0.0, 1.0)
    
    return calibrated_x, calibrated_y

def draw_calibration_screen(img, current_step, time_left):
    """Dibuja la pantalla de calibración"""
    h, w = img.shape[:2]
    
    # Fondo
    img.fill(0)
    
    if current_step < len(calibration_targets):
        target_x, target_y = calibration_targets[current_step]
        
        # Convertir a coordenadas de píxeles
        px = int(target_x * w)
        py = int(target_y * h)
        
        # Dibujar punto de calibración más grande
        cv2.circle(img, (px, py), 40, (0, 255, 255), -1)
        cv2.circle(img, (px, py), 50, (255, 255, 255), 4)
        
        # Animación de pulso
        pulse = int(20 * (1 + np.sin(time.time() * 5)))
        cv2.circle(img, (px, py), 50 + pulse, (255, 255, 0), 2)
        
        # Instrucciones más grandes
        cv2.putText(img, "CALIBRACION", (w//2 - 150, 60), FONT, 1.2, (255, 255, 0), 3)
        cv2.putText(img, "Mira al punto amarillo", (w//2 - 180, 100), FONT, 0.9, (255, 255, 255), 2)
        cv2.putText(img, f"Paso {current_step + 1}/{len(calibration_targets)}", (w//2 - 100, 140), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"Tiempo: {time_left}s", (w//2 - 80, 180), FONT, 0.8, (0, 255, 255), 2)
        
        # Dibujar todos los puntos de calibración
        for i, (tx, ty) in enumerate(calibration_targets):
            color = (0, 255, 0) if i < current_step else (100, 100, 100)
            size = 10 if i == current_step else 6
            cv2.circle(img, (int(tx * w), int(ty * h)), size, color, -1)

def draw_keyboard(img, out_text, sel_r, sel_c, show_calibration_info=False):
    """Dibuja el teclado con letras más grandes y mejor diseño"""
    # Calcular dimensiones del teclado
    kb_w = GRID_COLS * CELL_W + 2 * MARGIN
    kb_h = GRID_ROWS * CELL_H + 2 * MARGIN + 100  # Espacio extra para texto
    
    # Crear teclado con fondo más profesional
    kb = np.zeros((kb_h, kb_w, 3), dtype=np.uint8)
    
    # Fondo gradiente
    for i in range(kb_h):
        intensity = 30 + int(20 * (i / kb_h))
        kb[i, :] = (intensity, intensity, intensity)
    
    # Dibujar celdas
    idx = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x = MARGIN + c * CELL_W
            y = MARGIN + r * CELL_H
            letter = letters[idx]
            
            # Determinar color y estilo de la celda
            if r == sel_r and c == sel_c:
                # Celda seleccionada - resaltar
                color = (0, 200, 0)
                bg_color = (50, 100, 50)
                thickness = 4
                # Efecto de resaltado
                cv2.rectangle(kb, (x-2, y-2), (x+CELL_W+2, y+CELL_H+2), (0, 255, 0), 2)
            else:
                # Celda normal
                color = (200, 200, 200)
                bg_color = (60, 60, 60)
                thickness = 2
            
            # Fondo de celda
            cv2.rectangle(kb, (x, y), (x+CELL_W, y+CELL_H), bg_color, -1)
            # Borde de celda
            cv2.rectangle(kb, (x, y), (x+CELL_W, y+CELL_H), color, thickness)
            
            # Texto centrado - más grande y con sombra
            txt_size = cv2.getTextSize(letter, FONT, FONT_SCALE, 3)[0]
            tx = x + (CELL_W - txt_size[0]) // 2
            ty = y + (CELL_H + txt_size[1]) // 2
            
            # Sombra del texto
            cv2.putText(kb, letter, (tx+2, ty+2), FONT, FONT_SCALE, (0, 0, 0), 3, cv2.LINE_AA)
            # Texto principal
            cv2.putText(kb, letter, (tx, ty), FONT, FONT_SCALE, color, 3, cv2.LINE_AA)
            
            # Número pequeño en esquina (opcional)
            cv2.putText(kb, str(idx+1), (x+5, y+20), FONT, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
            
            idx += 1
    
    # Área de texto de salida - más grande y mejor diseñada
    output_bg_y = kb_h - 80
    cv2.rectangle(kb, (MARGIN, output_bg_y), (kb_w-MARGIN, kb_h-MARGIN), (40, 40, 40), -1)
    cv2.rectangle(kb, (MARGIN, output_bg_y), (kb_w-MARGIN, kb_h-MARGIN), (100, 100, 100), 2)
    
    # Texto de salida con scroll si es muy largo
    display_text = out_text if len(out_text) <= 25 else "..." + out_text[-22:]
    cv2.putText(kb, "Texto:", (MARGIN+10, output_bg_y - 10), FONT, 0.6, (200, 200, 200), 1)
    cv2.putText(kb, display_text, (MARGIN+15, output_bg_y + 30), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Información de estado
    status_y = 25
    cv2.putText(kb, "TECLADO OCULAR", (MARGIN, status_y), FONT, 0.7, (255, 255, 0), 2)
    
    # Información de calibración
    if show_calibration_info:
        cal_status = "CALIBRADO" if calibration_matrix is not None else "SIN CALIBRAR"
        color_status = (0, 255, 0) if calibration_matrix is not None else (0, 100, 255)
        cv2.putText(kb, f"Estado: {cal_status}", (kb_w - 200, status_y), FONT, 0.5, color_status, 1)
        
        # Barra de progreso de dwell
        if sel_r >= 0 and sel_c >= 0:
            progress_width = int(150 * (dwell_counter / DWELL_FRAMES))
            cv2.rectangle(kb, (kb_w - 200, status_y + 10), (kb_w - 200 + progress_width, status_y + 20), (0, 255, 255), -1)
            cv2.rectangle(kb, (kb_w - 200, status_y + 10), (kb_w - 50, status_y + 20), (100, 100, 100), 1)
    
    # Instrucciones
    instructions = [
        "Mira a una letra para seleccionarla",
        "Mantén la mirada para escribir",
        "'c'=Calibrar, 'r'=Reset, ESC=Salir"
    ]
    
    for i, instruction in enumerate(instructions):
        cv2.putText(kb, instruction, (MARGIN, kb_h - 80 + 15*i), FONT, 0.4, (150, 150, 150), 1)
    
    return kb

# ---------- Cámara ----------
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("No se pudo abrir la cámara. Asegúrate que esté conectada.")
    exit(1)

print("Instrucciones:")
print("- Presiona 'c' para iniciar calibración")
print("- Presiona 'r' para resetear calibración") 
print("- Mira las letras y mantén la mirada para seleccionar")
print("- Presiona ESC para salir")

# Ajustar tamaño de ventana
cv2.namedWindow("Eye Keyboard - Teclado Ocular", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Eye Keyboard - Teclado Ocular", 1600, 900)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # espejo para UX natural
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Manejar calibración
    if calibrating:
        current_time = time.time()
        time_in_step = current_time - calibration_start_time
        time_left = max(0, CALIBRATION_DURATION - int(time_in_step))
        
        # Dibujar pantalla de calibración
        calibration_frame = frame.copy()
        draw_calibration_screen(calibration_frame, calibration_step, time_left)
        
        # Recolectar datos de mirada durante la calibración
        if time_in_step > 1.0:  # Esperar 1 segundo antes de empezar a recolectar
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(150,150))
            if len(faces) > 0:
                face = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
                fx, fy, fw, fh = face
                
                roi_gray = gray[fy:fy+fh, fx:fx+fw]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,20))
                
                gaze_points = []
                for (ex,ey,ew,eh) in eyes[:2]:
                    eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                    res = detect_pupil(eye_gray)
                    if res:
                        nx, ny, _, _ = res
                        # Convertir a coordenadas relativas a la cara
                        global_x = (ex + nx * ew) / fw
                        global_y = (ey + ny * eh) / fh
                        gaze_points.append((global_x, global_y))
                
                if gaze_points:
                    avg_gaze = np.mean(gaze_points, axis=0)
                    calibration_data["gaze_points"].append(avg_gaze)
                    calibration_data["screen_points"].append(calibration_targets[calibration_step])
        
        # Avanzar al siguiente paso cuando se acabe el tiempo
        if time_in_step >= CALIBRATION_DURATION:
            calibration_step += 1
            calibration_start_time = current_time
            
            if calibration_step >= len(calibration_targets):
                calibrating = False
                update_calibration()
        
        # Mostrar solo la calibración
        cv2.imshow("Eye Keyboard - Teclado Ocular", calibration_frame)
        
    else:
        # Modo normal de funcionamiento
        face = None
        gaze_detected = False
        raw_gaze_x, raw_gaze_y = 0.5, 0.5  # Valores por defecto (centro)

        # Detectar cara (una sola)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(150,150))
        if len(faces) > 0:
            # tomar la cara más grande
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            (fx,fy,fw,fh) = faces[0]
            face = faces[0]
            cv2.rectangle(frame, (fx,fy), (fx+fw, fy+fh), (100,255,100), 2)

            # dentro de la cara, detectamos ojos (Haar)
            roi_gray = gray[fy:fy+fh, fx:fx+fw]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,20))

            pupils = []
            for (ex,ey,ew,eh) in eyes[:2]:
                pad = 3
                ex2 = max(ex-pad, 0)
                ey2 = max(ey-pad, 0)
                ew2 = min(ew+2*pad, fw - ex2)
                eh2 = min(eh+2*pad, fh - ey2)
                eye_gray = roi_gray[ey2:ey2+eh2, ex2:ex2+ew2]
                res = detect_pupil(eye_gray)
                if res:
                    nx, ny, th_img, (cx_local, cy_local) = res
                    pupil_x_global = fx + ex2 + cx_local
                    pupil_y_global = fy + ey2 + cy_local
                    pupils.append((pupil_x_global, pupil_y_global, nx, ny))
                    cv2.circle(frame, (pupil_x_global, pupil_y_global), 6, (0,255,255), -1)
                    cv2.circle(frame, (pupil_x_global, pupil_y_global), 2, (0,0,0), -1)

            # calcular posición de mirada
            if len(pupils) >= 1:
                gaze_detected = True
                avg_x = np.mean([p[0] for p in pupils])
                avg_y = np.mean([p[1] for p in pupils])

                fx,fy,fw,fh = face
                raw_gaze_x = (avg_x - fx) / float(fw)
                raw_gaze_y = (avg_y - fy) / float(fh)

                # suavizar
                smooth_queue_x.append(raw_gaze_x)
                smooth_queue_y.append(raw_gaze_y)
                sx = np.mean(smooth_queue_x)
                sy = np.mean(smooth_queue_y)

                # Aplicar calibración
                if calibration_matrix is not None:
                    calibrated_x, calibrated_y = apply_calibration(sx, sy)
                else:
                    calibrated_x, calibrated_y = sx, sy

                # mapear a celda del teclado
                col = int(calibrated_x * GRID_COLS)
                row = int(calibrated_y * GRID_ROWS)
                col = max(0, min(GRID_COLS-1, col))
                row = max(0, min(GRID_ROWS-1, row))

                # Debounce / dwell
                if (row, col) == last_cell:
                    dwell_counter += 1
                else:
                    dwell_counter = 0
                    last_cell = (row, col)

                # Seleccionar letra
                if dwell_counter >= DWELL_FRAMES:
                    idx = row*GRID_COLS + col
                    chosen = letters[idx]
                    if chosen.strip() != "":
                        output_text += chosen
                        print(f"Letra seleccionada: '{chosen}' - Texto: {output_text}")
                    dwell_counter = 0

                selected_row, selected_col = row, col

                # Mostrar información en frame de cámara
                info_text = f"Mirada: {calibrated_x:.2f},{calibrated_y:.2f}"
                cv2.putText(frame, info_text, (10, 30), FONT, 0.6, (255,255,0), 2)
                cv2.putText(frame, f"Celda: [{row},{col}]", (10, 60), FONT, 0.6, (255,255,0), 2)
                cv2.putText(frame, f"Progreso: {dwell_counter}/{DWELL_FRAMES}", (10, 90), FONT, 0.6, (255,255,0), 2)

        # Construir interfaz
        kb_img = draw_keyboard(frame, output_text, selected_row, selected_col, show_calibration_info=True)
        
        # Concatenar teclado y cámara
        h1, w1 = kb_img.shape[:2]
        h2, w2 = frame.shape[:2]
        
        # Redimensionar frame de cámara para que coincida en altura con teclado
        frame_resized = cv2.resize(frame, (w2, h1))
        
        # Concatenar
        canvas = np.hstack((kb_img, frame_resized))
        
        cv2.imshow("Eye Keyboard - Teclado Ocular", canvas)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC
        break
    elif k == ord('c') and not calibrating:  # Iniciar calibración
        start_calibration()
    elif k == ord('r'):  # Resetear calibración
        calibration_matrix = None
        output_text = ""
        dwell_counter = 0
        print("Calibración y texto reseteados")
    elif k == ord('\b') or k == 127:  # backspace
        output_text = output_text[:-1] if output_text else ""
    elif k == ord(' '):  # espacio
        output_text += ' '

cap.release()
cv2.destroyAllWindows()