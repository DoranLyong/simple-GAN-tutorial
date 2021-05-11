"""
(ref) https://github.com/DoranLyong/U-NET-tutorial/blob/main/U-NET_pytorch/utils.py
"""


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
    
):
    print("*** | Data loading... | ***")