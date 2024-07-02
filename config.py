batch_size = 120  # from paper GitHub code
epochs = 300
weights_ema_decay = 0.999
adam_params = {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-7, "weight_decay": 0}
model_args = [28, 28, 1, 10]
lr_decay = 0.98
aug_rot = 30
aug_erase = 4 / 28
aug_min_with = 21  # 0.75*28
experiment_path = "trained_model"
seed = 66451774
log_col_names = [
    "epoch",
    "train_acc",
    "train_loss",
    "test_acc",
    "test_loss",
    "test_ema_acc",
    "test_ema_loss",
]
