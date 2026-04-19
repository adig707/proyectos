import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import threading
import queue

CurrentFolder = os.getcwd()
image = CurrentFolder+'\\Alguien1.jpg'
image2 = CurrentFolder+'\\Alguien2.jpg'

if not os.path.exists(image) or not os.path.exists(image2):
    print("Error: Imágenes no encontradas")
    exit()

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)

if not video_capture.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

person1_image = face_recognition.load_image_file(image)
person1_encoding = face_recognition.face_encodings(person1_image)[0]
person2_image = face_recognition.load_image_file(image2)
person2_encoding = face_recognition.face_encodings(person2_image)[0]

known_encodings = np.array([person1_encoding, person2_encoding])
known_names = ["Alguien1", "Alguien2"]

face_locations = []
face_names = []
already_attendence = set()
process_counter = 0
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue()

def process_face_recognition():
    while True:
        if not frame_queue.empty():
            small_frame = frame_queue.get()
            
            locations = face_recognition.face_locations(small_frame, number_of_times_to_upsample=1, model="hog")
            
            if locations:
                encodings = face_recognition.face_encodings(small_frame, locations, num_jitters=1)
                
                names = []
                for encoding in encodings:
                    distances = np.linalg.norm(known_encodings - encoding, axis=1)
                    min_distance_idx = np.argmin(distances)
                    
                    if distances[min_distance_idx] < 0.5:
                        names.append(known_names[min_distance_idx])
                    else:
                        names.append("Desconocido")
                
                result_queue.put((locations, names))

recognition_thread = threading.Thread(target=process_face_recognition, daemon=True)
recognition_thread.start()

colors = {"Alguien1": (0, 255, 0), "Alguien2": (255, 0, 0), "Desconocido": (0, 0, 255)}

print("Sistema iniciado. Presiona 'q' para salir")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    process_counter += 1
    
    if process_counter % 5 == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        if not frame_queue.full():
            frame_queue.put(rgb_small_frame)
    
    if not result_queue.empty():
        face_locations, face_names = result_queue.get()
        
        for name in face_names:
            if name != "Desconocido" and name not in already_attendence:
                print(f"✓ {name} detectado - {datetime.now().strftime('%H:%M:%S')}")
                already_attendence.add(name)
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        color = colors.get(name, (128, 128, 128))
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(frame, f"Detectados: {len(already_attendence)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('Reconocimiento Facial', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
print(f"Personas detectadas: {list(already_attendence)}")