import cv2
import numpy as np
import os 
import time
import torch
import torch.nn as nn
import torchvision 
from torchvision import models, transforms
import os
import sys
import tkinter as tk
from PIL import Image, ImageTk

def fix_russian_text(text):
    """Конвертирует UTF-8 строки в кодировку Windows для OpenCV"""
    return text.encode('utf-8').decode('cp1251')

MODEL_PATH = "pelmeni_model.pth"

CLASS_NAMES = ['кипящая_вода','кипящие_пельмени','нейтральная_сцена']

CONFIDENCE_THRESHOLD = 0.80

IMAGE_SIZE = 224

CHECK_EVERY_N_FRAMES = 15

TIMER_MNIUTES = 8

RTSP_TIMEOUT = 2500
#Загрузка нейросети 
os.system('chcp 866 >nul 2>&1')

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
wait_until = 0.0 

last_timer_print_time = 0

frame_counter = 0

print("\n Слежу за водой\n")

root = tk.Tk()
root.title("Pelmeni Scaner-AI ver1.048597 BETA")
root.geometry("1040x620")

canvas = tk.Canvas(root, width=640, height=480, bg="white")
canvas.pack(side="left", fill="y", expand=False)

right_panel = tk.Frame(root, width=400, bg="#17001E")
right_panel.pack(side="right", fill="both", expand=True)


status_var = tk.StringVar(value="Слежу за водой")
status_label = tk.Label(
    right_panel,
    textvariable=status_var,
    font=("Fixedsys Excelsior 3.01", 14),
    fg="#00ff00",
    bg="#17001E",
    justify="left",
    anchor="nw",
    wraplength=380
)
status_label.pack(side="top", fill="x", padx=10, pady=10)

pole = tk.Entry(right_panel, width=50, bg="#17001E", fg="#00FF00", insertbackground="#00FF00")
pole.insert(0, "rtsp://192.168.0.10:1642")
pole.pack(padx=10, pady=4)

label_min = tk.Label(right_panel, text="Минут варки:", font=("Consolas", 12), fg="#00FF00", bg="#17001E")
label_min.pack(padx=10, pady=4)

krutilka = tk.Spinbox(right_panel, from_=1, to=30, width=10, bg="#CA92FF", fg="#00FF00")
krutilka.delete(0, "end")
krutilka.insert(0, "8")
krutilka.pack(padx=10, pady=10)

TEKUSCHY_ADRES = "rtsp://192.168.0.10:1642"
UZHE_RABOTAET = False

cap = None

def ya_skazala_startuem():
    global TEKUSCHY_ADRES, TIMER_MNIUTES, UZHE_RABOTAET, cap
    if UZHE_RABOTAET:
        status_var.set("Уже запущено")
        return

    adres = pole.get().strip()
    if not adres.startswith("rtsp://"):
        status_var.set("Адрес должен начинаться с rtsp://")
        return

    try:
        minutu = int(krutilka.get())
    except ValueError:
        status_var.set("ВВЕДИ ЧИСЛО А НЕ ИЕРОГИЛФЫ")
        return

    TEKUSCHY_ADRES = adres
    TIMER_MNIUTES = minutu

    print(f"подключение к серверу на телефоне: {TEKUSCHY_ADRES}")
    status_var.set(f"Подключаюсь к {TEKUSCHY_ADRES}...")

    cap = cv2.VideoCapture(TEKUSCHY_ADRES, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_TIMEOUT)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_TIMEOUT)

    if not cap.isOpened():
        status_var.set("Неверный адрес или не запущена камера")
        cap.release()
        return

    ret, frame = cap.read()
    if not ret or frame is None:
        status_var.set("Первый кадр стал лост медией")
        cap.release()
        return

    UZHE_RABOTAET = True
    status_var.set("ПОДКЛЮЧЕНО СЛЕЖУ ЗА ВОДОЙ")
    root.after(33, tick)

    
knopka = tk.Button(right_panel, text="НАЧАТЬ СКАНИРОВАНИЕ ПЕЛЬМЕНЕЙ", command=ya_skazala_startuem)
knopka.pack(padx=10, pady=10)

def tick():
    global frame_counter, current_state, timer_start, last_timer_print_time, wait_until, cap
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Потерян кадр")
        cap.release()
        cap = cv2.VideoCapture(TEKUSCHY_ADRES, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        root.after(5000, tick)
        return

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    black = np.mean(gray_frame) < 10.0
    if black:
        display_label = "ЧЁРНЫЙ ЭКРАН ИЛИ НЕТ СИГНАЛА Я ХЗ"
        

    if not black:
        frame_counter += 1
        if frame_counter % CHECK_EVERY_N_FRAMES == 0 and current_state != State_Gotovo:
            class_name, confidence = predict(frame)

            if confidence >= CONFIDENCE_THRESHOLD:
                status_var.set(f"Вижу: {class_name} | Уверенность: {confidence:.0%}")
                print(f"Вижу: {class_name} | Уверенность: {confidence:.0%}")
            else:
                status_var.set(f"Сомнение: {class_name} | Уверенность: {confidence:.0%} (ниже порога {CONFIDENCE_THRESHOLD})")
                print(f"Сомнение: {class_name} | Уверенность: {confidence:.0%} (ниже порога {CONFIDENCE_THRESHOLD})")

            if current_state == State_Wath_Water:
                if class_name == 'кипящая_вода' and confidence >= CONFIDENCE_THRESHOLD:
                    print("\nВОДА ЗАКИПЕЛА\n")
                    current_state = State_Wath_Pelmeni
                    wait_until = time.time() + 75

            elif current_state == State_Wath_Pelmeni:
                if time.time() >= wait_until \
                    and class_name == 'кипящие_пельмени' and  confidence >= CONFIDENCE_THRESHOLD:
                    print("ПИЛЬМЕНИ КИПЯТ НАХУЙ")
                    timer_start = time.time()
                    current_state = State_Wath_Varka

            elif current_state == State_Wath_Varka:
                elapsed = time.time() - timer_start
                remaining = (TIMER_MNIUTES * 60) - elapsed
                if remaining <= 0:
                        print("ПЕЛЬМЕНИ ГОТОВЫ")
                        current_state = State_Gotovo
                else:
                    now = time.time()
                    if now - last_timer_print_time >= 60:
                        mins_left = int(remaining // 60)
                        secs_left = int(remaining % 60)
                        print(f"Осталось: {mins_left} мин, {secs_left} сек")
                        last_timer_print_time = now
                        mins_left = int(remaining // 60)
                        secs_left = int(remaining % 60)
                        status_var.set(f"Ещё нужно варить: {int(remaining // 60)}:{int(remaining % 60):02d}")

            elif current_state == State_Gotovo: 
                 status_var.set("ПЕЛЬМЕНИ ГОТОВЫ!")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(Image.fromarray(rgb))
    canvas.delete("cam")
    canvas.create_image(0,0, anchor="nw", image=photo, tags="cam")
    canvas.photo = photo

    root.after(33, tick)
    
def on_close():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
