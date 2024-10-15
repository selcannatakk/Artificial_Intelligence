import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np



class YOLOv3(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(1024 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 7 * 7 * (5 * 5 + num_classes))

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = F.max_pool2d(F.relu(self.conv5(x)), 2)
        x = F.max_pool2d(F.relu(self.conv6(x)), 2)
        x = F.max_pool2d(F.relu(self.conv7(x)), 2)
        x = x.view(x.size(0), 1024 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx]).replace('\\','/')
        image = Image.open(img_name).convert("RGB")

        label_name = os.path.join(self.label_dir, self.images[idx].replace(".jpg", ".txt")).replace('\\','/')
        with open(label_name, 'r') as f:
            label = f.readline().strip().split()  # Örnek bir etiket formatı, gerektiğinde özelleştirilebilir

        if self.transform:
            image = self.transform(image)

        return image, label



# Eğitim için parametreler
image_dir = "./datasets/license_plate_dataset/train/images"  # Görüntülerin bulunduğu dizin
label_dir = "./datasets/license_plate_dataset/train/labels"  # Etiket dosyalarının bulunduğu dizin
num_classes = 1  # Sınıf sayısı (örneğin, sadece araba için 1)
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Veri augmentasyonu ve normalizasyonu
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Veri setini yükleme
custom_dataset = CustomDataset(image_dir=image_dir, label_dir=label_dir, transform=transform)
data_loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True)


# Modeli ve kayıp fonksiyonunu tanımlama
model = YOLOv3(num_classes)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Eğitim döngüsü
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        optimizer.zero_grad()
        outputs = model(images)
        # Çıktıları ve etiketleri kayıp fonksiyonuna gönderme (düzeltme gerekebilir)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(data_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Eğitim sonrası modeli kaydetme
torch.save(model.state_dict(), "yolov3_custom_model.pth")
print("Model saved successfully!")
