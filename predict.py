import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import time
import sys

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc(x)
        return x

def prepare_image(image_path):
    img = Image.open(image_path).convert('L')
    
    img = ImageOps.invert(img)
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    tensor_img = transform(img).unsqueeze(0)
    return tensor_img

def predict(image_path, model_path='mnist_cnn.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CNN().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Успішно завантажено модель з '{model_path}'")
    except FileNotFoundError:
        print(f"Помилка: файл '{model_path}' не знайдено!")
        return

    model.eval()

    try:
        input_tensor = prepare_image(image_path).to(device)
    except FileNotFoundError:
        print(f"Помилка: зображення '{image_path}' не знайдено!")
        return

    start_time = time.perf_counter()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    end_time = time.perf_counter()

    execution_time = (end_time - start_time) * 1000 

    print("\n--- Результати розпізнавання ---")
    print(f"Передбачена цифра: {prediction.item()}")
    print(f"Впевненість моделі: {confidence.item() * 100:.2f}%")
    print(f"Час розпізнавання: {execution_time:.3f} мс")

if __name__ == '__main__':
    image_file = 'my_digit.png' 
    predict(image_file)