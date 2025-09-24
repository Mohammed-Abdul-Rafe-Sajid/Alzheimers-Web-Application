import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ----------------------------
# Model Definition (must match training!)
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # 3 channels (RGB input)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # For 224x224 input → after 3 pools → 28x28 feature maps
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)   # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----------------------------
# Image Transform (same as training)
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def preprocess_image(uploaded_file):
    """Preprocess uploaded image for model prediction"""
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")   # ensure 3 channels
    return transform(image).unsqueeze(0)  # add batch dimension

# ----------------------------
# Model Loader
# ----------------------------
def load_model(model_path, num_classes=4):
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# ----------------------------
# Prediction Function
# ----------------------------

def predict(model, image_tensor, class_names=None):
    """Run prediction on a preprocessed image tensor"""
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        if class_names:
            return class_names[predicted.item()]
        return predicted.item()

