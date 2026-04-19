import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import threading
import queue
import pandas as pd
import glob
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PersonDatabase:
    """Gestiona la base de datos de rostros conocidos"""
    
    def __init__(self, photos_folder="photos"):
        self.photos_folder = Path(photos_folder)
        self.known_encodings = []
        self.known_names = []
        self.encoding_lock = threading.Lock()
        
        self.photos_folder.mkdir(exist_ok=True)
        logger.info(f"✓ Carpeta '{photos_folder}' verificada")
        
        self.load_all_faces()
    
    def load_all_faces(self):
        """Carga todas las caras desde la carpeta de fotos"""
        with self.encoding_lock:
            self.known_encodings = []
            self.known_names = []
            
            # Soportar múltiples formatos
            patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            image_files = []
            for pattern in patterns:
                image_files.extend(self.photos_folder.glob(pattern))
            
            for image_path in image_files:
                try:
                    image = face_recognition.load_image_file(str(image_path))
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        name = image_path.stem
                        self.known_encodings.append(encodings[0])
                        self.known_names.append(name)
                        logger.info(f"✓ Cargado: {name}")
                    else:
                        logger.warning(f"⚠️ No se detectó cara en: {image_path.name}")
                        
                except Exception as e:
                    logger.error(f"❌ Error cargando {image_path.name}: {e}")
            
            if self.known_encodings:
                self.known_encodings = np.array(self.known_encodings)
                logger.info(f"✓ Base de datos cargada: {len(self.known_names)} personas")
            else:
                logger.warning("⚠️ No se encontraron fotos válidas")
    
    def add_person_photo(self, frame, name):
        """Añade una nueva foto a la base de datos"""
        if not name or not name.strip():
            logger.error("Nombre vacío proporcionado")
            return False
        
        name = name.strip()
        filepath = self.photos_folder / f"{name}.jpg"
        
        try:
            # Guardar imagen
            cv2.imwrite(str(filepath), frame)
            logger.info(f"✓ Foto guardada: {filepath}")
            
            # Procesar encoding
            image = face_recognition.load_image_file(str(filepath))
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                logger.warning("⚠️ No se detectó cara en la nueva foto")
                return False
            
            # Actualizar base de datos de forma thread-safe
            with self.encoding_lock:
                if len(self.known_encodings) == 0:
                    self.known_encodings = np.array([encodings[0]])
                else:
                    self.known_encodings = np.vstack([self.known_encodings, encodings[0]])
                self.known_names.append(name)
            
            logger.info(f"✓ {name} agregado a la base de datos")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error guardando/procesando foto: {e}")
            return False
    
    def get_encodings_copy(self):
        """Retorna una copia thread-safe de los encodings"""
        with self.encoding_lock:
            return self.known_encodings.copy() if len(self.known_encodings) > 0 else None, self.known_names.copy()


class ExcelManager:
    """Gestiona el registro de asistencia en Excel"""
    
    def __init__(self, filename="asistencia.xlsx"):
        self.filename = filename
        self.lock = threading.Lock()
        
        if os.path.exists(filename):
            try:
                self.df = pd.read_excel(filename)
                if 'Estado' not in self.df.columns:
                    self.df['Estado'] = 'Presente'
            except Exception as e:
                logger.error(f"Error cargando Excel existente: {e}")
                self.df = pd.DataFrame(columns=['Nombre', 'Fecha', 'Hora', 'Estado'])
        else:
            self.df = pd.DataFrame(columns=['Nombre', 'Fecha', 'Hora', 'Estado'])
    
    def add_person(self, name, is_new_registration=False):
        """Registra una persona en el Excel"""
        now = datetime.now()
        status = "Nuevo Registro" if is_new_registration else "Presente"
        
        new_row = {
            'Nombre': name,
            'Fecha': now.strftime('%Y-%m-%d'),
            'Hora': now.strftime('%H:%M:%S'),
            'Estado': status
        }
        
        with self.lock:
            self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
            self.save()
        
        emoji = "🆕" if is_new_registration else "✓"
        logger.info(f"{emoji} {name} - {status} - {now.strftime('%H:%M:%S')}")
    
    def save(self):
        """Guarda el DataFrame en Excel"""
        try:
            self.df.to_excel(self.filename, index=False)
        except Exception as e:
            logger.error(f"Error guardando Excel: {e}")


class FaceRecognitionSystem:
    """Sistema principal de reconocimiento facial"""
    
    def __init__(self):
        self.person_db = PersonDatabase()
        self.excel_manager = ExcelManager()
        
        # Estados del sistema
        self.already_attendance = set()
        self.newly_registered = set()
        self.name_colors = {}
        self.colors_list = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        
        # Colas para comunicación entre threads
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.excel_queue = queue.Queue()
        self.photo_queue = queue.Queue()
        
        # Variables de control
        self.current_frame = None
        self.face_locations = []
        self.face_names = []
        self.running = True
        
        # Iniciar threads
        self.start_threads()
    
    def start_threads(self):
        """Inicia los threads de procesamiento"""
        threads = [
            threading.Thread(target=self.process_face_recognition, daemon=True),
            threading.Thread(target=self.process_excel, daemon=True),
            threading.Thread(target=self.process_photos, daemon=True)
        ]
        for t in threads:
            t.start()
    
    def process_face_recognition(self):
        """Thread para procesar reconocimiento facial"""
        while self.running:
            try:
                if not self.frame_queue.empty():
                    small_frame = self.frame_queue.get(timeout=0.1)
                    
                    # Detectar ubicaciones de caras
                    locations = face_recognition.face_locations(
                        small_frame,
                        number_of_times_to_upsample=1,
                        model="hog"
                    )
                    
                    if not locations:
                        continue
                    
                    # Obtener encodings de forma thread-safe
                    known_encodings, known_names = self.person_db.get_encodings_copy()
                    
                    if known_encodings is None or len(known_encodings) == 0:
                        # No hay base de datos, marcar todos como desconocidos
                        names = ["Desconocido"] * len(locations)
                        confidences = [1.0] * len(locations)
                    else:
                        # Calcular encodings de las caras detectadas
                        encodings = face_recognition.face_encodings(small_frame, locations, num_jitters=1)
                        
                        names = []
                        confidences = []
                        
                        for encoding in encodings:
                            # Calcular distancias usando vectorización
                            distances = np.linalg.norm(known_encodings - encoding, axis=1)
                            min_idx = np.argmin(distances)
                            min_distance = distances[min_idx]
                            
                            # Umbral de reconocimiento
                            if min_distance < 0.5:
                                names.append(known_names[min_idx])
                                confidences.append(min_distance)
                            else:
                                names.append("Desconocido")
                                confidences.append(min_distance)
                    
                    # Enviar resultados
                    if not self.result_queue.full():
                        self.result_queue.put((locations, names, confidences))
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error en reconocimiento: {e}")
    
    def process_excel(self):
        """Thread para procesar registros de Excel"""
        while self.running:
            try:
                if not self.excel_queue.empty():
                    data = self.excel_queue.get(timeout=0.1)
                    if isinstance(data, tuple):
                        name, is_new = data
                        self.excel_manager.add_person(name, is_new)
                    else:
                        self.excel_manager.add_person(data, False)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error en Excel: {e}")
    
    def process_photos(self):
        """Thread para procesar nuevas fotos"""
        while self.running:
            try:
                if not self.photo_queue.empty():
                    frame, name = self.photo_queue.get(timeout=0.1)
                    success = self.person_db.add_person_photo(frame, name)
                    if success:
                        self.newly_registered.add(name)
                        logger.info(f"✅ {name} listo para detección")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error procesando foto: {e}")
    
    def get_person_name(self):
        """Solicita el nombre de una nueva persona"""
        print("\n=== CAPTURA DE FOTO ===")
        print("La persona debe estar mirando a la cámara")
        name = input("Nombre de la persona: ").strip()
        return name if name else None
    
    def draw_faces(self, frame):
        """Dibuja rectángulos y nombres en las caras detectadas"""
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Escalar coordenadas
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Asignar color
            if name not in self.name_colors:
                color_idx = len(self.name_colors) % len(self.colors_list)
                self.name_colors[name] = self.colors_list[color_idx]
            
            color = self.name_colors.get(name, (128, 128, 128))
            
            # Dibujar rectángulo y nombre
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    def draw_info(self, frame):
        """Dibuja información del sistema en el frame"""
        info_text = f"BD: {len(self.person_db.known_names)} | Asistencia: {len(self.already_attendance)} | Nuevos: {len(self.newly_registered)}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        controls_text = "'f'=Foto  'r'=Recargar  'q'=Salir"
        cv2.putText(frame, controls_text, (10, 460),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Ejecuta el sistema principal"""
        # Inicializar cámara
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
        
        if not video_capture.isOpened():
            logger.error("No se pudo abrir la cámara")
            return
        
        logger.info("=== SISTEMA DE RECONOCIMIENTO FACIAL ===")
        logger.info("Controles: 'q'=Salir | 'f'=Foto | 'r'=Recargar")
        
        frame_counter = 0
        
        try:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break
                
                self.current_frame = frame.copy()
                frame_counter += 1
                
                # Procesar cada 5 frames
                if frame_counter % 5 == 0:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    if not self.frame_queue.full():
                        self.frame_queue.put(rgb_small_frame)
                
                # Obtener resultados de reconocimiento
                if not self.result_queue.empty():
                    self.face_locations, self.face_names, _ = self.result_queue.get()
                    
                    # Registrar asistencia
                    for name in self.face_names:
                        if name != "Desconocido" and name not in self.already_attendance:
                            self.already_attendance.add(name)
                            is_new = name in self.newly_registered
                            self.excel_queue.put((name, is_new))
                            
                            emoji = "🆕" if is_new else "🎯"
                            logger.info(f"{emoji} DETECCIÓN: {name}")
                
                # Dibujar caras e información
                self.draw_faces(frame)
                self.draw_info(frame)
                
                cv2.imshow('Sistema de Reconocimiento Facial', frame)
                
                # Manejo de teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('f') and self.current_frame is not None:
                    cv2.putText(frame, "FOTO CAPTURADA - Ingresa nombre", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow('Sistema de Reconocimiento Facial', frame)
                    cv2.waitKey(100)
                    
                    name = self.get_person_name()
                    if name:
                        self.photo_queue.put((self.current_frame, name))
                        
                elif key == ord('r'):
                    logger.info("Recargando base de datos...")
                    self.person_db.load_all_faces()
        
        finally:
            self.running = False
            video_capture.release()
            cv2.destroyAllWindows()
            
            logger.info(f"Personas en asistencia: {list(self.already_attendance)}")
            logger.info(f"Nuevas registradas: {list(self.newly_registered)}")
            logger.info(f"Total en BD: {len(self.person_db.known_names)}")


if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()