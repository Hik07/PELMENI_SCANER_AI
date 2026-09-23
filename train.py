import torch
import torchvision
from torchvision import datasets, transforms, models 
import torch.nn as nn 
import torch.optim as optim
import os

#НАСТРОЙКИ 

DATASET_DIR = "dataset"

MODEL_SAVE_PATH = "pelmeni_model.pth"

EPOCHS = 20

BATCH_SIZE = 16

IMAGE_SIZE = 224

LEARNING_RATE = 0.001

# ПОДОТОВКА ДАННЫХ 

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    
    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(),

    transforms.ToTensor(),    

    transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])

])

dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transforms)

print(f"Классы: {dataset.classes}")

NUM_CLASSES = len(dataset.classes)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Используеться: {device}")

model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(device)

#параметры обучения

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#ОБУЧЕНИЕ 

print("НАЧАЛОСЬ ОБУЧЕНИЕ")

for epoch in range(EPOCHS):
    
    model.train()
    
    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()

        total += labels.size(0)

    accuracy = 100 * correct / total

    avg_loss = running_loss / len(dataloader)

    print(f"Эпохуй [{epoch+1}/{EPOCHS}] Ошибка: {avg_loss:.4f} Точность: {accuracy:.1f}%")

#СОХРАНЕНИЕ МОДЕЛИ 

torch.save(model.state_dict(), MODEL_SAVE_PATH)

print(f"\n✅ Модель сохранена в файл: {MODEL_SAVE_PATH}")
print(f"Порядок классов: {dataset.classes}")
print("Запиши этот порядок! Он нужен в server.py")