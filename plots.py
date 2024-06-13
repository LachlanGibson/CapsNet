import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from model import CapsNet


def read_csv(file_path, col_names):
    def to_num(s):
        if "." in s or "e" in s:
            return float(s)
        else:
            return int(s)

    with open(file_path, "r") as file:
        lines = file.readlines()
    rows = [line.strip().split(",") for line in lines]
    cols = [list(col) for col in zip(*rows)]
    cols = [list(map(to_num, col)) for col in cols]
    data = {name: col for col, name in zip(cols, col_names)}
    return data


log_path = "./checkpoints_augmented/log.csv"
fig_path = "./figures"
col_names = ["epoch", "avg_loss", "avg_norm", "test_acc", "test_loss"]

data = read_csv(log_path, col_names)
test_error_rate = [1 - acc for acc in data["test_acc"]]

print("maximum test accuracy:", max(data["test_acc"]))
print("final test error rate:", data["test_acc"][-1])
print("minimum test loss:", min(data["test_loss"]))
print("final test loss:", data["test_loss"][-1])


# plot loss curves
plt.plot(data["epoch"], data["avg_loss"], label="Average training loss")
plt.plot(data["epoch"], data["test_loss"], label="Test loss")
plt.legend(loc="upper right")
plt.yscale("log")
plt.title("Loss curves")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.savefig(os.path.join(fig_path, "loss.svg"), bbox_inches="tight")
plt.savefig(os.path.join(fig_path, "loss.png"), bbox_inches="tight")
plt.close()

# plot test accuracy
plt.plot(data["epoch"], data["test_acc"], label="Test accuracy")
plt.title("Test accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(fig_path, "test_accuracy.svg"), bbox_inches="tight")
plt.savefig(os.path.join(fig_path, "test_accuracy.png"), bbox_inches="tight")
plt.close()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CapsNet(28, 28, 1, 10).to(device)
model.load_state_dict(torch.load("./checkpoints_augmented/checkpoint300.pth"))
model.eval()

test_data = datasets.MNIST(
    root="", train=False, download=True, transform=transforms.ToTensor()
)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# find misclassified images
def find_errors(imgs, targets):
    imgs, targets = imgs.to(device), targets.to(device)
    with torch.no_grad():
        output = model(imgs)
    predicted = output.argmax(dim=-1)
    inds = predicted != targets
    misclassified = imgs[inds]
    probs = output[inds]
    targets = targets[inds]
    return [misclassified, probs, targets]


misclassified, probs, targets = tuple(
    zip(*[find_errors(imgs, targets) for imgs, targets in test_loader])
)

misclassified = torch.cat(misclassified)
probs = 100 * torch.softmax(torch.cat(probs), -1)
targets = torch.cat(targets)

num_rows = int((misclassified.shape[0] - 1) ** 0.5) + 1
num_cols = num_rows
plt.figure(figsize=(2 * num_rows, 2 * num_cols))
for i in range(num_rows * num_cols):
    if i >= misclassified.shape[0]:
        break
    plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(misclassified[i].squeeze().cpu().numpy(), cmap="gray")
    pred = probs[i].argmax().item()
    targ = targets[i].item()
    plt.title(
        f"{pred}: {probs[i,pred].round().int().item()}%,  {targ}: {probs[i,targ].round().int().item()}%"
    )
    plt.axis("off")
plt.savefig(os.path.join(fig_path, "misclassified.svg"), bbox_inches="tight")
plt.savefig(os.path.join(fig_path, "misclassified.png"), bbox_inches="tight")
plt.close()
