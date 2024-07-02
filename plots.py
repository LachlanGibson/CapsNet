import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torchvision import datasets

from config import *
from model import HVCapsNet


def read_csv(file_path):
    def to_num(s):
        if "." in s or "e" in s:
            return float(s)
        else:
            return int(s)

    with open(file_path, "r") as file:
        lines = file.readlines()
    rows = [line.strip().split(",") for line in lines]
    col_names = rows.pop(0)
    cols = [list(col) for col in zip(*rows)]
    cols = [list(map(to_num, col)) for col in cols]
    data = {name: col for col, name in zip(cols, col_names)}
    return data


# create figures directory
fig_path = os.path.join(experiment_path, "figures")
if os.path.exists(fig_path):
    raise FileExistsError("figures path already exists")
os.makedirs(fig_path)

# read log file
log_path = os.path.join(experiment_path, "log.csv")
data = read_csv(log_path)

# print final results
print("maximum test accuracy:", max(data["test_ema_acc"]))
print("final test error rate:", data["test_ema_acc"][-1])
print("minimum test loss:", min(data["test_ema_loss"]))
print("final test loss:", data["test_ema_loss"][-1])


# plot loss curves
plt.plot(data["epoch"], data["train_loss"], label="Average training loss")
plt.plot(data["epoch"], data["test_loss"], label="Test loss")
plt.plot(data["epoch"], data["test_ema_loss"], label="EMA test loss")
plt.legend(loc="upper right")
plt.yscale("log")
plt.title("Loss curves")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.savefig(os.path.join(fig_path, "loss.svg"), bbox_inches="tight")
plt.savefig(os.path.join(fig_path, "loss.png"), bbox_inches="tight")
plt.close()

# plot test accuracy
plt.plot(data["epoch"], data["train_acc"], label="Average training accuracy")
plt.plot(data["epoch"], data["test_acc"], label="Test accuracy")
plt.plot(data["epoch"], data["test_ema_acc"], label="EMA test accuracy")
plt.legend(loc="lower right")
plt.title("Test accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(fig_path, "test_accuracy.svg"), bbox_inches="tight")
plt.savefig(os.path.join(fig_path, "test_accuracy.png"), bbox_inches="tight")
plt.close()

# load model and test data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multi_avg_fn = get_ema_multi_avg_fn(weights_ema_decay)
model_ema = AveragedModel(HVCapsNet(*model_args), multi_avg_fn=multi_avg_fn).to(device)
model_ema.load_state_dict(
    torch.load(os.path.join(experiment_path, "model_ema.pth"), map_location=device)
)
model_ema.eval()
test_data = datasets.MNIST(
    root="", train=False, download=True, transform=transforms.ToTensor()
)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# find misclassified images
def find_errors(imgs, targets):
    imgs, targets = imgs.to(device), targets.to(device)
    with torch.no_grad():
        output = model_ema(imgs)
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

# plot misclassified images
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
