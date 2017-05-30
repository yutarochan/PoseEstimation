'''
Auxilary Functions for Image Processing and Plotting
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

# Plot Properties
sns.set_style("whitegrid", {'axes.grid' : False})

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

def plot_heatmap(im, hm, alpha=0.65):
    ''' Plot Probability Heatmap for All Joint Keypoints '''
    hm = np.swapaxes(hm, 0, 2)
    for i in range(19):
        plt.imshow(hm[i].T, cmap='jet_r')
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), alpha=alpha)
    plt.show()

def plot_joint_heatmap(im, hm, index, alpha=0.65):
    ''' Plot Probability Heatmap for Single Selected Joint Keypoint '''
    hm = np.swapaxes(hm, 0, 2)
    plt.imshow(hm[index].T, cmap='jet')
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), alpha=alpha)
    plt.show()

def plot_all_keypoints(im, pk, pts_size=4):
    pts = [j for i in pk for j in i]
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    for i in pts: plt.plot(i[0], i[1], 'ro', ms=pts_size)
    plt.show()

def plot_idx_keypoints(im, pk, index, pts_size=4):
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    for i in pk[index]: plt.plot(i[0], i[1], 'ro', ms=pts_size)
    plt.show()

def plot_paf(im, paf, index, alpha=0.65):
    paf = np.swapaxes(paf, 0, 2)
    plt.imshow(paf[index].T, cmap='jet')
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), alpha=alpha)
    plt.show()

def plot_person(im, point, limb, index, stick_width):
    for i in range(17):
        index = subset[index][np.array()]
