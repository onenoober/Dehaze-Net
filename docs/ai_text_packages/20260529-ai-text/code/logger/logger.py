from matplotlib import pyplot as plt
import numpy as np
import os


def plot_loss_log(loss_log, epoch, loss_dir):
    for key in loss_log.keys():
        values = np.array(loss_log[key])
        axis = np.arange(1, len(values) + 1)
        label = '{} Loss'.format(key)
        fig = plt.figure()
        plt.title(label)
        plt.plot(axis, values, label=label)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(loss_dir, 'loss_{}.pdf'.format(key)))
        plt.close(fig)


def plot_psnr_log(psnr_log, epoch, psnr_dir):
    values = np.array(psnr_log)
    axis = np.arange(1, len(values) + 1)
    label = 'PSNR'
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, values, label=label)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig(os.path.join(psnr_dir, 'psnr.pdf'))
    plt.close(fig)
