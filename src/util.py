'''
Auxilary Functions for Image Processing and Plotting
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
import cv2
import numpy as np
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter

# Handle for No-Display
# TODO: Enable no - display mode for servers.
try:
    import matplotlib.pyplot as plt
except Exception as e:
    pass

import model

# Plot Properties
sns.set_style("whitegrid", {'axes.grid' : False})

# Load Model JSON File
model_file = '../model/pose_model.json'
m = model.ModelData(model_file)

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

def plot_person(im, data, pts_size=5, stick_width=4):
    # print(data[0])

    for idx, d in enumerate(data):
        # Plot Base Image
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

        # Plot Joint Keypoints
        for i in range(17):
            if d[i]: plt.plot(d[i][0], d[i][1], 'o', ms=pts_size, color=np.array(m.model['colors'][idx])/255)

        # Plot Limbs
        for i in range(len(m.model['limbSeq'][:-2])):
            idx = (np.array(m.model['limbSeq'][i]) - 1)
            # Check if non existant or special cases...
            if not d[idx[0]] or not d[idx[1]]: continue
            print(idx)
            X = [d[idx[0]][0], d[idx[1]][0]]
            Y = [d[idx[0]][1], d[idx[1]][1]]
            p1 = d[idx[0]][0:2]
            p2 = d[idx[1]][0:2]
            print(p1, p2)

            plt.plot(X, Y, 'k-')
            # index = d[i][np.array(m.model['limbSeq'][i])-1]
    plt.show()
    '''
    for i in range(17):
        index = subset[index][np.array()]
    '''
