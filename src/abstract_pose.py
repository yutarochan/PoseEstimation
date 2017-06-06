'''
Abstract Pose Modeling
Prototype Test for Generating Pose Abstraction from a Given the Set of Keypoints

Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import os
import cv2
import json
import numpy as np

import util

# Import Pose Data
data = '../data/pose_data/000015.json'
im_path = '../data/test/000015.jpg'
pose = json.loads(open(data, 'rb').read())['pose']

''' Abstraction Process '''
abs_limbMap = [0, 1, [3, 6], [4, 7], [8, 11], [9, 12], [10, 13]]
abs_limbSeq = [[0, 1], [1, 2], [2, 3], [1, 4], [4, 5], [5, 6]]

# Work with Only One Person For Testing
pose = pose[0]

# Generate Abstract Body Part Keypoints
abs_limb = []
for limb in abs_limbMap:
    if type(limb) is int and pose[limb] is not None:
        abs_limb.append(pose[limb])
    elif type(limb) is list:
        if pose[limb[0]] is not None and pose[limb[1]] is not None:
            abs_limb.append([(pose[limb[0]][0] + pose[limb[1]][0])/2, (pose[limb[0]][1] + pose[limb[1]][1])/2])
        elif pose[limb[0]] is not None:
            abs_limb.append([pose[limb[0]][0], pose[limb[0]][1]])
        elif pose[limb[1]] is not None:
            abs_limb.append([pose[limb[1]][0], pose[limb[1]][1]])
        else:
            abs_limb.append(None)
    else:
        abs_limb.append(None)

# Fit a Projection Line

print('Abstract Joint Coordinates:')
print(str(abs_limb), '\n')

# Plot Abstract Points
im = cv2.imread(im_path)
util.plot_abs_pose(im, abs_limb)
