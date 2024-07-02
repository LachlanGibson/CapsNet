import os
import shutil

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from augmentation import compose_transforms
from config import *
from model import HVCapsNet

# create experiment directory and copy over config
if os.path.exists(experiment_path):
    raise FileExistsError("experiment path already exists")
os.makedirs(experiment_path)
shutil.copy("config.py", os.path.join(experiment_path, "config.py"))

# set seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# derived variables
log_path = os.path.join(experiment_path, "log.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create data loaders
transform = compose_transforms(aug_rot, aug_min_with, aug_erase)
training_data = datasets.MNIST(root="", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="", train=False, download=True, transform=ToTensor())
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# create model, optimiser, scheduler, criterion
model = HVCapsNet(*model_args).to(device)
multi_avg_fn = get_ema_multi_avg_fn(weights_ema_decay)
model_ema = AveragedModel(model, multi_avg_fn=multi_avg_fn)
optimiser = torch.optim.Adam(model.parameters(), **adam_params)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, lr_decay)
criterion = torch.nn.CrossEntropyLoss()


def save_model(model, file_name):
    torch.save(
        model.state_dict(),
        os.path.join(experiment_path, file_name),
    )


def evaluate(model, loader):
    model.eval()
    correct, total, loss = 0, 0, 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        with torch.no_grad():
            output = model(imgs)
            predicted = output.argmax(dim=-1)
            total += targets.shape[0]
            correct += (predicted == targets).sum().item()
            loss += targets.shape[0] * criterion(output, targets).item()
    return correct / total, loss / total


def update_bn(loader, model):
    model.train()
    for imgs, _ in loader:
        imgs = imgs.to(device)
        model(imgs)


print(
    "number of learnable parameters",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)

with open(log_path, "a") as f:
    f.write(",".join(log_col_names) + "\n")
for epoch in range(epochs):
    cum_loss = 0
    cum_acc = 0
    for i, (imgs, targets) in enumerate(train_loader):
        model.train()
        imgs, targets = imgs.to(device), targets.to(device)
        optimiser.zero_grad()
        output = model(imgs)
        loss = criterion(output, targets)
        loss.backward()
        cum_acc += (output.argmax(dim=-1) == targets).sum().item()
        cum_loss += targets.shape[0] * loss.item()
        optimiser.step()
        model_ema.update_parameters(model)
    scheduler.step()

    update_bn(train_loader, model_ema)
    train_acc, train_loss = cum_acc / len(training_data), cum_loss / len(training_data)
    test_acc, test_loss = evaluate(model, test_loader)
    test_ema_acc, test_ema_loss = evaluate(model_ema, test_loader)
    with open(log_path, "a") as f:
        f.write(
            f"{epoch+1},{train_acc},{train_loss},{test_acc},{test_loss},{test_ema_acc},{test_ema_loss}\n"
        )
    print(
        f"Epoch {epoch+1}/{epochs} | train, test, ema test | acc: {train_acc:.4f}, {test_acc:.4f}, {test_ema_acc:.4f} | loss: {train_loss:.4f}, {test_loss:.4f}, {test_ema_loss:.4f}"
    )

save_model(model, "model.pth")
save_model(model_ema, "model_ema.pth")
