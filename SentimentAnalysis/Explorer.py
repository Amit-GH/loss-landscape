import numpy as np
import matplotlib.pyplot as plt


class Explorer:
    def __init__(self):
        pass

    def plot_loss_curve(self, x_epochs, y_loss, save_fig=False, filename_suffix="", directory_path="plots/"):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x_epochs, y_loss, color='red', label='Average Loss')
        ax.legend()
        ax.set(
            xlabel='Epochs',
            ylabel='Average Loss',
            title="Average loss vs epochs",
        )
        if save_fig:
            plt.savefig(directory_path + "loss_curve_" + filename_suffix + ".jpg", format='jpg', dpi=300)
        else:
            plt.show()


    def plot_train_test_curves_for(self, x, y_tr, y_te, save_fig=False, filename_suffix="", directory_path="plots/"):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, y_tr, color='red', label='Average loss for train data')
        ax.plot(x, y_te, color='blue', label='Average loss for validation data')
        ax.legend()
        ax.set(
            xlabel='Epochs',
            ylabel='Average Loss',
            title="Average Loss for train and validation data for multiple epochs",
        )
        if save_fig:
            plt.savefig(directory_path + "train_test_curve_" + filename_suffix + ".jpg", format='jpg', dpi=300)
        else:
            plt.show()

