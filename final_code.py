import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
#-----------------------------------------------------------------dataset--------------------------------------------------------
class FaceForensicsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with subfolders 'real' and 'fake'.
            transform (callable, optional): Transformations to apply to images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        for label, class_name in enumerate(["real", "fake"]):  
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"warning:{class_dir} does not exist!")
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_path.endswith((".jpg",",png",".jpeg")):
                    self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert("RGB")  # Convert to RGB format
        except Exception as e:
            print(f"error opening image {img_path}:{e}" )
            return None, None
        if self.transform:
            image = self.transform(image)

        return image, label

#----------------------------------------------------------------data preprocessing-------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
dataset = FaceForensicsDataset(root_dir="C:/Users/anjis/OneDrive/Desktop/faceforencis_real", transform=transform)
print("total samples in dataset: ", len(dataset))
dataset.data = [(img, label) for img, label in dataset.data if img is not None] 
print("Total samples after filtering:", len(dataset))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


#----------------------------------------------------model-------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(8192, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

class CompactEnsembleDiscriminator(nn.Module):
    def __init__(self, num_models=3):
        super(CompactEnsembleDiscriminator, self).__init__()
        self.models = nn.ModuleList([Discriminator() for _ in range(num_models)])
        self.fc = nn.Linear(512*num_models, 1)  

    def forward(self, x):
        outputs = [model(x) for model in self.models]  
        outputs = torch.cat(outputs, dim=1)  # Concatenate along the channel dimension
        outputs = outputs.view(outputs.size(0), -1) # Flatten the output from each model
        outputs = self.fc(outputs)  
        return torch.sigmoid(outputs)

# ----------------------------------------------------model, Loss & Optimizer-----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CompactEnsembleDiscriminator().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

#-----------------------------------------training the discriminator---------------------------------------------
num_epochs = 10  
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device).float().view(-1, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if (i+1) % 10 == 0: # Print loss every 10 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

#------------------------------------------------------------results--------------------------------------------
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.cpu().numpy()
        outputs = model(images).cpu().numpy()
        predictions = (outputs > 0.5).astype(int)
        
        y_true.extend(labels)
        y_pred.extend(predictions)

# Convert to NumPy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute Metrics
accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
try: 
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
except ValueError:
    print("Confusion matrix could not be calculated (likely only one class present).")
    sensitivity = 0.0 

# Print Results
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"F1 Score: {f1:.4f}")    