import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# Constants
root_dir = 'data/hymenoptera_data/train'
root_val = 'data/hymenoptera_data/val'
ants_label_dir = 'ants'
bees_label_dir = 'bees'
batch_size = 64
epochs = 40

# Transformer
transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Label to number mapping
label_to_num = {"ants": 0, "bees": 1}

# Dataset class
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.listdir(self.path)

    def __getitem__(self, idx):
        image_name = self.image_path[idx]
        image = Image.open(os.path.join(self.path, image_name)).convert('RGB')
        label = self.label_dir
        return transformer(image), label_to_num[label]

    def __len__(self):
        return len(self.image_path)

# Datasets and DataLoaders
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)
all_dataset = ants_dataset + bees_dataset
dataloader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True)

ants_val = MyData(root_val, ants_label_dir)
bees_val = MyData(root_val, bees_label_dir)
all_val = ants_val + bees_val
val_loader = DataLoader(all_val, batch_size=batch_size, shuffle=True)

# Model
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 7, 1, 3),
            nn.MaxPool2d(4),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(4),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)

# Training setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
my_model = MyModule().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001, weight_decay=0.01)

# Tensorboard writer
writer = SummaryWriter("my_dl_logs")

train_step=0
total_test_step=0

# Training loop
for i in range(epochs):
    print('epoch', i + 1)
    my_model.train()

    for data in dataloader:
        image, label = data
        image, label = image.to(device), label.to(device)
        outputs = my_model(image)

        loss = loss_fn(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_step += 1
        writer.add_scalar('train loss', loss.item(), train_step)
        if train_step % 10 == 0:
            print(f"step:{train_step} train loss:{loss.item():.4f}")

    my_model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in val_loader:
            image, label = data
            image, label = image.to(device), label.to(device)
            outputs = my_model(image)
            loss = loss_fn(outputs, label)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == label).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss / len(val_loader)))
    print("整体测试集上的正确率：{}".format(total_accuracy / len(val_loader)))
    writer.add_scalar("test_loss", total_test_loss / len(val_loader), total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / len(val_loader), total_test_step)
    total_test_step += 1

writer.close()
