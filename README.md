# CapsNet reproduction

In this project I attempted to reproduce some of the results outlined in "[No Routing Needed Between Capsules](https://doi.org/10.48550/arXiv.2001.09136)" by Adam Byerly, Tatiana Kalganova, and Ian Dear first published in January 2020 at arXiv:2001.09136 \[cs.CV\]. I reproduced the network architecture that utilised Z-derived capsules where the branch logits are merged using a sum without any learnable parameters.

The network was trained following Byerly, et al. for 300 epochs with a batch size of 120 with the Adam optimiser using the same learning rate schedule and other hyper parameters. The final test accuracy was measured using the exponential moving averaged model weights. The only difference I can identify is that I updated the EMA model batchnorm layers on the training data after every epoch, not just at the end of training.

## Results

Training the network as described above resulted in a final test prediction accuracy of 99.62% and a maximum of 99.66% during training, with prediction losses of about 0.01239 and 0.01206 respectively. This result is little lower than the reported min accuracy of 99.74%. I am yet to identify the cause of this discrepency.

Here are the misclassified digits with corresponding targets and probabilities.

![alt text](https://github.com/LachlanGibson/CapsNet/blob/main/trained_model/figures/misclassified.png?raw=true)

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
