import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import RandomAffine, RandomApply, RandomRotation

from model import CapsNet

batch_size = 128
epochs = 300
lr = 1e-3
wd = 1e-5
lr_decay = 0.98
grad_norm_clip = 2
aug_rot = 30
aug_trans = 2 / 28
aug_erase = 4 / 28
aug_scale = 0.75
augment_data = True
checkpoint_path = "checkpoints_augmented"
log_path = os.path.join(checkpoint_path, "log.csv")
checkpoint_interval = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if augment_data:
    transform = transforms.Compose(
        [
            RandomApply([RandomRotation(aug_rot)], p=0.5),
            RandomApply([RandomAffine(0, translate=(aug_trans, 0))], p=0.5),
            RandomApply([RandomAffine(0, translate=(0, aug_trans))], p=0.5),
            RandomAffine(0, scale=(aug_scale, 1)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=1, scale=(aug_erase, aug_erase), ratio=(1, 1)),
        ]
    )
else:
    transform = transforms.ToTensor()

training_data = datasets.MNIST(root="", train=True, download=True, transform=transform)

test_data = datasets.MNIST(
    root="", train=False, download=True, transform=transforms.ToTensor()
)

train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

torch.manual_seed(7164)
if torch.cuda.is_available():
    torch.cuda.manual_seed(7164)


for i, (imgs, targets) in enumerate(train_loader):
    plt.figure(figsize=(10, 10))
    for j in range(25):
        plt.subplot(5, 5, j + 1)
        plt.imshow(imgs[j].squeeze().numpy(), cmap="gray")
        plt.title(targets[j].item())
        plt.axis("off")
    plt.savefig("./figures/augmented.svg", bbox_inches="tight")
    plt.savefig("./figures/augmented.png", bbox_inches="tight")
    plt.close()
    break

model = CapsNet(28, 28, 1, 10).to(device)
optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, lr_decay)
criterion = torch.nn.CrossEntropyLoss()
print(
    "number of learnable parameters",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    cum_loss = 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        with torch.no_grad():
            output = model(imgs)
        predicted = output.argmax(dim=-1)
        total += targets.shape[0]
        correct += (predicted == targets).sum().item()
        cum_loss += targets.shape[0] * criterion(output, targets).item()
    return correct / total, cum_loss / total


torch.save(model.state_dict(), os.path.join(checkpoint_path, "checkpoint0.pth"))
for epoch in range(epochs):
    cum_loss = 0
    cum_norm = 0
    for i, (imgs, targets) in enumerate(train_loader):
        model.train()
        imgs, targets = imgs.to(device), targets.to(device)
        optimiser.zero_grad()
        output = model(imgs)
        loss = criterion(output, targets)
        loss.backward()
        cum_loss += loss.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        cum_norm += norm
        optimiser.step()
    scheduler.step()
    test_acc, test_loss = evaluate(model, test_loader)
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(
                f"{epoch+1},{cum_loss/len(train_loader)},{cum_norm/len(train_loader)},{test_acc},{test_loss}\n"
            )
    print(
        f"Epoch {epoch+1}/{epochs} | avg loss: {cum_loss/len(train_loader):.4f} | avg norm: {cum_norm/len(train_loader):.4f} | test acc: {test_acc:.4f}"
    )
    if (epoch + 1) % checkpoint_interval == 0:
        torch.save(
            model.state_dict(),
            os.path.join(checkpoint_path, f"checkpoint{epoch+1}.pth"),
        )
