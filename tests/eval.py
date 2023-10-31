import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

import internimage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imagenet_path = "./dataset/ImageNet/"

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

dataset = torchvision.datasets.ImageNet(imagenet_path, split="val", transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

model = internimage.create_model("internimage_t_1k_224", pretrained=True).to(device)
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy_top1 = 100 * correct / total

print(f"Accuracy top1: {accuracy_top1}")
