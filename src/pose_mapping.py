'''
Temporal Pose Mapping
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import cv2
import json
import pprint
from scipy.spatial import KDTree

import util

# Load Frame Files
frame_1 = '../data/000001.json'
frame_2 = '../data/000002.json'

im_1 = '../data/000001.jpg'
im_2 = '../data/000002.jpg'

f1 = json.loads(open(frame_1, 'rb').read())['pose']
f2 = json.loads(open(frame_2, 'rb').read())['pose']

# Construct 2D-KDTree Mapping of Joints
kd1_frame = []
kp1_lookup = []
for i in range(18):
    X = []
    Y = []
    lookup = dict()

    for j in range(len(f1)):
        X.append(f1[j][i][0])
        Y.append(f1[j][i][1])
        lookup[tuple(f1[j][i])] = j

    kd1_frame.append(KDTree(zip(X, Y)))
    kp1_lookup.append(lookup)

# util.plot_person(cv2.imread(im_1), f1)

# Perform Lookup on Each Point in Next Frame
kp2_lookup = []
for i in range(18):
    lookup = dict()
    for j in range(len(f2)):
        res = kd1_frame[i].query(f2[j][i])
        lookup[tuple(f2[j][i])] = kp1_lookup[0][tuple(kd1_frame[0].data[res[1]])]
    kp2_lookup.append(lookup)

# Remap Lookup to List
f2_mapped = [[] for i in range(len(f2))]
for i, kp in enumerate(kp2_lookup):
    for j in kp.keys(): f2_mapped[kp2_lookup[i][j]].append(j)
    # for i in kp.iterkeys(): f2_mapped[kp2_lookup(i)] = i

# util.plot_person(cv2.imread(im_1), f2_mapped)
'''
pp = pprint.PrettyPrinter()
pp.pprint(f2_mapped)
# pp.pprint(kp2_lookup)
'''
