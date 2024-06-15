# CapsNet reproduction

In this project I attempted to reproduce some of the results outlined in "[No Routing Needed Between Capsules](https://doi.org/10.48550/arXiv.2001.09136)" by Adam Byerly, Tatiana Kalganova, and Ian Dear first published in January 2020 at arXiv:2001.09136 \[cs.CV\]. I reproduced the network architecture that utilised Z-derived capsules where the branch logits are merged using a sum without any learnable parameters.

The network was initialised using PyTorch defaults and trained using the AdamW optimiser (rather than Adam). I followed the same learning rate schedule, but did use weight decay (1e-5) and clipped gradient norms to 2. Data was augmented the same way, except translations were done randomly up to 2px and rather than adjusting image widths, the images were scaled, maintaining their original aspect ratio.

## Results

Training the network as described above resulted in a final test prediction accuracy of 99.66% and prediction loss of about 0.010077, a little lower than the reported min of 99.74%. This discrepency might be explained by any combination of the changes outlined above.

Here are the misclassified digits with corresponding targets and probabilities.

![alt text](https://github.com/LachlanGibson/CapsNet/blob/main/figures/misclassified.png?raw=true)

## Installation

To install this project, follow these steps:

1. Clone the repository.
2. Run `pip install -r requirements.txt` to install the dependencies.
3. Use Python version 3.11 or later.

## Usage

I followed these steps:

1. coded the network modules in `model.py`
2. trained the model by running `mnist_train.py`
3. produced figures by running `plots.py`

## License

This project is licensed under the [MIT License](https://github.com/LachlanGibson/CapsNet/blob/main/LICENSE).
