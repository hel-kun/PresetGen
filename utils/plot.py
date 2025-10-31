import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def plot_loss_curve(train_losses, val_losses, save_path: Optional[str]=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    if save_path:
        plt.savefig(save_path)
        plt.close()