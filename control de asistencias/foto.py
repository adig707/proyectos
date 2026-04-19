import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import threading
import queue
import pandas as pd
import glob

class PersonDatabase:
    def __init__(self, photos_folder="photos"):
        self.photos_folder = photos_folder
        self.known_encodings = []
        self.known_names = []
        
        if not os.path.exists(photos_folder):
            os.makedirs(photos_folder)
            print(f"✓ Carpeta '{photos_folder}' creada")
        
        self.load_all_faces()
    
    def load_all_faces(self):
        self.known_encodings = []
        self.known_names = []
        
        image_files = glob.glob(os.path.join(self.photos_folder, "*.jpg")) + \
                     glob.glob(os.path.join(self.photos_folder, "*.jpeg")) + \
                     glob.glob(os.path.join(self.photos_folder, "*.png"))
        
        for image_path in image_files:
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    filename = os.path.basename(image_path)
                    name = os.path.splitext(filename)[0]
                    
                    self.known_encodings.append(encodings[0])
                    self.known_names.append(name)
                    print(f"✓ Cargado: {name}")
                else:
                    print(f"⚠️ No se detectó cara en: {image_path}")
                    
            except Exception as e:
                print(f"❌ Error cargando {image_path}: {e}")
        
        if self.known_encodings:
            self.known_encodings = np.array(self.known_encodings)
            print(f"✓ Base de datos cargada: {len(self.known_names)} personas")
        else:
            print("⚠️ No se encontraron fotos válidas en la carpeta 'photos'")
    
    def add_person_photo(self, frame, name):
        if not name or name.strip() == "":
            return False
        
        filename = f"{name.strip()}.jpg"
        filepath = os.path.join(self.photos_folder, filename)
        
        try:
            cv2.imwrite(filepath, frame)
            print(f"✓ Foto guardada: {filepath}")
            
            # Cargar solo esta nueva persona para detección inmediata
            try:
                image = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    self.known_encodings = np.append(self.known_encodings, [encodings[0]], axis=0)
                    self.known_names.append(name.strip())
                    print(f"✓ {name.strip()} agregado a la base de datos (detección inmediata)")
                    return True
                else:
                    print(f"⚠️ No se detectó cara en la nueva foto")
                    
            except Exception as e:
                print(f"❌ Error procesando nueva foto: {e}")
                # Si falla, recargar toda la base de datos
                self.load_all_faces()
            
            return True
        except Exception as e:
            print(f"❌ Error guardando foto: {e}")
            return False

class ExcelManager:
    def __init__(self, filename="asistencia.xlsx"):
        self.filename = filename
        self.df = pd.DataFrame(columns=['Nombre', 'Fecha', 'Hora', 'Estado'])
        if os.path.exists(filename):
            try:
                self.df = pd.read_excel(filename)
                # Si no tiene la columna Estado, agregarla
                if 'Estado' not in self.df.columns:
                    self.df['Estado'] = 'Presente'
            except:
                pass
    
    def add_person(self, name, is_new_registration=False):
        now = datetime.now()
        status = "Nuevo Registro" if is_new_registration else "Presente"
        
        new_row = {
            'Nombre': name,
            'Fecha': now.strftime('%Y-%m-%d'),
            'Hora': now.strftime('%H:%M:%S'),
            'Estado': status
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self.save()
        
        status_emoji = "🆕" if is_new_registration else "✓"
        print(f"{status_emoji} {name} registrado en Excel como '{status}' - {now.strftime('%H:%M:%S')}")
    
    def save(self):
        try:
            self.df.to_excel(self.filename, index=False)
        except Exception as e:
            print(f"Error guardando Excel: {e}")

def get_person_name():
    print("\n=== CAPTURA DE FOTO ===")
    print("La persona debe estar mirando a la cámara")
    name = input("Nombre de la persona: ").strip()
    return name if name else None

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)

if not video_capture.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

person_db = PersonDatabase()
excel_manager = ExcelManager()

face_locations = []
face_names = []
already_attendence = set()
newly_registered = set()  # Nuevas personas registradas en esta sesión
process_counter = 0
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue()
excel_queue = queue.Queue()
photo_queue = queue.Queue()
current_frame = None
capture_photo = False

def process_face_recognition():
    while True:
        if not frame_queue.empty():
            small_frame = frame_queue.get()
            
            locations = face_recognition.face_locations(small_frame, number_of_times_to_upsample=1, model="hog")
            
            if locations and len(person_db.known_encodings) > 0:
                encodings = face_recognition.face_encodings(small_frame, locations, num_jitters=1)
                
                names = []
                confidences = []
                for encoding in encodings:
                    distances = np.linalg.norm(person_db.known_encodings - encoding, axis=1)
                    min_distance_idx = np.argmin(distances)
                    min_distance = distances[min_distance_idx]
                    
                    if min_distance < 0.5:
                        names.append(person_db.known_names[min_distance_idx])
                        confidences.append(min_distance)
                    else:
                        names.append("Desconocido")
                        confidences.append(min_distance)
                
                result_queue.put((locations, names, confidences))
            elif locations:
                # Si hay caras pero no hay base de datos
                names = ["Desconocido"] * len(locations)
                confidences = [1.0] * len(locations)
                result_queue.put((locations, names, confidences))

def process_excel():
    while True:
        if not excel_queue.empty():
            data = excel_queue.get()
            if isinstance(data, tuple):
                name, is_new = data
                excel_manager.add_person(name, is_new)
            else:
                # Compatibilidad con formato anterior
                excel_manager.add_person(data, False)

def process_photos():
    while True:
        if not photo_queue.empty():
            frame, name = photo_queue.get()
            person_db.add_person_photo(frame, name)

recognition_thread = threading.Thread(target=process_face_recognition, daemon=True)
excel_thread = threading.Thread(target=process_excel, daemon=True)
photo_thread = threading.Thread(target=process_photos, daemon=True)
recognition_thread.start()
excel_thread.start()
photo_thread.start()

colors_list = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
name_colors = {}

print("=== SISTEMA DE RECONOCIMIENTO FACIAL ===")
print("Controles:")
print("  'q' - Salir")
print("  'f' - Capturar foto de nueva persona")
print("  'r' - Recargar base de datos")
print("Sistema iniciado...")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    current_frame = frame.copy()
    process_counter += 1
    
    if process_counter % 5 == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        if not frame_queue.full():
            frame_queue.put(rgb_small_frame)
    
    if not result_queue.empty():
        face_data = result_queue.get()
        
        if len(face_data) == 3:
            face_locations, face_names, face_confidences = face_data
        else:
            # Compatibilidad con formato anterior
            face_locations, face_names = face_data
            face_confidences = [0.0] * len(face_names)
        
        for name in face_names:
            if name != "Desconocido" and name not in already_attendence:
                already_attendence.add(name)
                
                # Verificar si es una persona recién registrada
                is_new_registration = name in newly_registered
                excel_queue.put((name, is_new_registration))
                
                if is_new_registration:
                    print(f"🆕 NUEVA PERSONA REGISTRADA: {name} - Primera detección automática")
                else:
                    print(f"🎯 DETECCIÓN: {name} registrado en asistencia")
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        if name not in name_colors:
            color_idx = len(name_colors) % len(colors_list)
            name_colors[name] = colors_list[color_idx]
        
        color = name_colors.get(name, (128, 128, 128))
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(frame, f"BD: {len(person_db.known_names)} | Asistencia: {len(already_attendence)} | Nuevos: {len(newly_registered)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "'f'=Foto 'r'=Recargar 'q'=Salir", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if capture_photo:
        cv2.putText(frame, "CAPTURANDO FOTO...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow('Sistema de Reconocimiento Facial', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        if current_frame is not None:
            capture_photo = True
            cv2.putText(frame, "FOTO CAPTURADA - Ingresa nombre en consola", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow('Sistema de Reconocimiento Facial', frame)
            cv2.waitKey(100)
            
            name = get_person_name()
            if name:
                # Procesar la foto inmediatamente
                success = person_db.add_person_photo(current_frame, name)
                if success:
                    newly_registered.add(name)  # Marcar como nueva persona registrada
                    print(f"✅ {name} agregado exitosamente - ¡Ya disponible para detección!")
                    print(f"📝 Próxima detección será marcada como 'Nuevo Registro' en Excel")
                    # Limpiar cache de detección para forzar nueva detección
                    while not result_queue.empty():
                        result_queue.get()
                else:
                    print(f"❌ Error procesando foto de {name}")
            capture_photo = False
    elif key == ord('r'):
        print("Recargando base de datos...")
        person_db.load_all_faces()

video_capture.release()
cv2.destroyAllWindows()
print(f"Personas en asistencia: {list(already_attendence)}")
print(f"Nuevas personas registradas: {list(newly_registered)}")
print(f"Total en base de datos: {len(person_db.known_names)}")