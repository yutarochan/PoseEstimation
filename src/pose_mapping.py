'''
Temporal Pose Mapping
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import os
import sys
import cv2
import json
import pprint
from os import listdir
from scipy.spatial import KDTree
from os.path import isfile, join

import util

# Multiframe Mapping
pose_dir = '../data/pose_data/'
pose = sorted([f for f in listdir(pose_dir) if isfile(join(pose_dir, f)) and f.split('.')[-1] == 'json'])

pose_map = dict()   # Dictionary to keep track of individual entities in frame.
entities = []       # Index frame in dictionary to track entity id.
ent_indx = -1       # Last added entity index.

# Following to Keep Track from Previous Frame
kd_frame  = []      # KD-Tree Pose Tree List
kp_lookup = []      # Key Point Lookup Dictionary List

for i in range(len(pose))[:3]:
    # Load Frame Data
    frame = json.loads(open(pose_dir+'/'+pose[i], 'rb').read())['pose']

    # Handle Entity Tracking Cases
    if len(entities) == 0:                  # No person in previous frame (previous frame was empty)
        print('Adding New Entities...')

        # Map Pose Dictionary Indicies
        for i in range(len(frame)):
            ent_indx += 1
            entities.append(ent_indx)
            pose_map[ent_indx] = []
            pose_map[ent_indx].append([tuple(i) for i in frame[i]])
            print('\t> Append Entity ID: ', ent_indx)

    elif len(frame) == len(entities):       # Same count as previous frame
        print('No Changes in Entity Count...')

        # Build Lookup Mapping for Current Frame
        curr_lookup = []
        for i in range(18):
            lookup = dict()
            for j in range(len(frame)):
                res = kd_frame[i].query(frame[j][i])
                lookup[tuple(frame[j][i])] = kp_lookup[0][tuple(kd_frame[0].data[res[1]])]
            curr_lookup.append(lookup)

        # Remap Lookup to List
        mapped = [[] for i in range(len(frame))]
        for i, kp in enumerate(curr_lookup):
            for j in kp.keys(): mapped[curr_lookup[i][j]].append(j)

        # Append Mapped Entirs to Pose Map Dictionary
        for i, p in enumerate(mapped): pose_map[i].append(p)

    elif len(frame) > len(entities):        # New person in frame
        print("Someone new came in...")

    elif len(frame) < len(entities):        # Person left frame
        print("Someone just left")

    # Construct 2D-KD Tree Mapping of Joints
    for i in range(18):
        X = []
        Y = []
        lookup = dict()

        for j in range(len(frame)):
            X.append(frame[j][i][0])
            Y.append(frame[j][i][1])
            lookup[tuple(frame[j][i])] = j

        kd_frame.append(KDTree(zip(X, Y)))
        kp_lookup.append(lookup)

    print('Entities on Frame: ', str(entities))
    print()

# Pretty Print Utility
# pp = pprint.PrettyPrinter()
# pp.pprint(pose_map)
