import torch
import torchvision.transforms as transforms
from torchvision.transforms import RandomApply, RandomRotation


class RandomWidthCompression(torch.nn.Module):
    def __init__(self, min_width: int):
        super(RandomWidthCompression, self).__init__()
        self.min_width = min_width  # minimum width after compression

    def forward(self, img):
        h, w_full = img.shape[-2:]
        new_w = torch.randint(self.min_width, w_full + 1, (1,)).item()
        compressed_img = transforms.functional.resize(img, (h, new_w))
        pad_w = w_full - new_w
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded_img = transforms.functional.pad(
            compressed_img, (pad_left, 0, pad_right, 0), fill=0
        )
        return padded_img

    def __repr__(self):
        return self.__class__.__name__ + f"(min_width={self.min_width})"


class RandomTranslate(torch.nn.Module):
    def __init__(self, dim: int, threshold: float = 1e-8):
        super(RandomTranslate, self).__init__()
        assert 0 <= threshold < 1
        assert dim in [0, 1]
        self.dim = dim
        self.threshold = threshold

    def forward(self, img):
        x = img.sum(0)
        used_pixels = x.gt(self.threshold)
        used_pixels = used_pixels.any(dim=1 - self.dim)
        used_pixels = used_pixels.nonzero().squeeze()
        if used_pixels.numel() == 0:
            return img
        left = used_pixels.min().item()
        right = x.shape[self.dim] - used_pixels.max().item()  # one more than the index
        shift = torch.randint(-left, right, (1,)).item()
        return torch.roll(img, shifts=shift, dims=self.dim + 1)

    def __repr__(self):
        return self.__class__.__name__ + f"({self.dim},{self.threshold})"


def compose_transforms(max_rotation, min_width, erase_scale):
    return transforms.Compose(
        [
            RandomApply([RandomRotation(max_rotation)], p=0.5),
            transforms.ToTensor(),
            RandomApply([RandomTranslate(0)], p=0.5),
            RandomApply([RandomTranslate(1)], p=0.5),
            RandomWidthCompression(min_width),
            transforms.RandomErasing(
                p=1, scale=(erase_scale, erase_scale), ratio=(1, 1)
            ),
        ]
    )
