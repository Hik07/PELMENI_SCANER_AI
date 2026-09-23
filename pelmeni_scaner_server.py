import cv2
import numpy as np
import os 
import time
import torch
import torch.nn as nn
import torchvision 
from torchvision import models, transforms

#Настройки 

MODEL_PATH = "pelmeni_model.pth"

CLASS_NAMES = ['кипящая_вода','кипящие_пельмени','нейтральная_сцена']

CONFIDENCE_THRESHOLD = 0.80

IMAGE_SIZE = 224

CHECK_EVERY_N_FRAMES = 15

TIMER_MNIUTES = 8

RTSP_URL = "rtsp://192.168.0.36:8080"

#Загрузка нейросети 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Используеться устройство: {device}")

model = models.resnet18(weights=None)

model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.to(device)

model.eval()

print("Нейроночка загруженна")

#Трансформация

preprocess = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(frame):

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_temsor = preprocess(rgb_frame)

    input_batch = input_temsor.unsqueeze(0)

    input_batch = input_batch.to(device)

    with torch.no_grad():
        output = model(input_batch)
        probailities = torch.softmax(output, dim=1)

        confidense, predicted_idx = torch.max(probailities, 1)

        class_name = CLASS_NAMES[predicted_idx.item()]

        return class_name, confidense.item()
    
State_Wath_Water = 1
State_Wath_Pelmeni = 2
State_Wath_Varka = 3
State_Gotovo = 4

current_state = State_Wath_Water

timer_start = None

last_timer_print_time = 0

frame_counter = 0


print("подключение к серверу на телефоне")

while True:
    cap = cv2.VideoCapture(RTSP_URL)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if cap.isOpened():

        ret, frame = cap.read()

        if ret and frame is not None:
            print("Подключенно к серверу!")
            break
        
        cap.release()

        print("Переподключение через 10 секунд...")
        time.sleep(10)

cv2.namedWindow("Камера с телефона", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Камера с телефона", 640, 480)

cv2.setWindowProperty("Камера с телефона", cv2.WND_PROP_TOPMOST, 1)

print("\n Слежу за водой\n")


while True:
    ret, frame = cap.read()

    if not ret:
        print("Потерян кадр")
        break
    frame_counter += 1

    display_label = ""

    display_color = (255,255,255)

    if frame_counter % CHECK_EVERY_N_FRAMES == 0:
        if current_state != State_Gotovo:
            class_name, confidence = predict(frame)

            display_label = f"{class_name} ({confidence:.0%})"

            print(f"Вижу: {class_name} | Уверенность: {confidence:.0%}")

            if current_state == State_Wath_Water:
                if class_name == 'кипящая_вода' and confidence >= CONFIDENCE_THRESHOLD:
                    print("\nВОДА ЗАКИПЕЛА\n")
                    current_state = State_Wath_Pelmeni

                    display_color = (0, 165, 255)

                    time.sleep(75)

            elif current_state == State_Wath_Pelmeni:
                if class_name == 'кипящие_пельмени' and  confidence >= CONFIDENCE_THRESHOLD:
                    print("ПИЛЬМЕНИ КИПЯТ НАХУЙ")
                    timer_start = time.time()
                    current_state = State_Wath_Varka

                    display_color = (0, 255, 0)

                elif current_state == State_Wath_Varka:

                    elapsed = time.time() - timer_start

                    remaining = (TIMER_MNIUTES * 60) - elapsed

                    if remaining <= 0:
                        print("ПЕЛЬМЕНИ ГОТОВЫ")
                        current_state = State_Gotovo
                        display_label = "ПЕЛЬМЕНИ ГОТОВЫ"
                        display_color = (0,255,0)
                    else:
                        now = time.time()

                        if now - last_timer_print_time >= 60:
                            mins_left = int(remaining // 60)
                            secs_left = int(remaining % 60)

                            print(f"Осталось: {mins_left} мин, {secs_left} сек")

                            last_timer_print_time = now

                            mins_left = int(remaining // 60)
                            secs_left = int(remaining % 60)

                            display_label = f"Варитья {mins_left}:{secs_left:02d}"

                            display_color = (0,255,255)

                    elif current_state == State_Gotovo:
                            
                        display_label = "ПЕЛЬМЕНИ ГОТОВЫ!"
                        display_color = (0,255,0)
    if display_label:
        cv2.putText(frame, display_label, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, display_color, 2)
    cv2.imshow("Камера с телефона", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break                

cap.release()
cv2.destroyAllWindows()
    
