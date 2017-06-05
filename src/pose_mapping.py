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

'''
# Load Frame Files
frame_1 = '../data/pose_data/000001.json'
frame_2 = '../data/pose_data/000002.json'

im_1 = '../data/test/000001.jpg'
im_2 = '../data/test/000002.jpg'

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

util.plot_person(cv2.imread(im_1), f1)

# Perform Lookup on Each Point in Next Frame
kp2_lookup = []
for i in range(18):
    lookup = dict()
    for j in range(len(f2)):
        # TODO: Handle case with a score threshold.
        res = kd1_frame[i].query(f2[j][i])
        lookup[tuple(f2[j][i])] = kp1_lookup[0][tuple(kd1_frame[0].data[res[1]])]
    kp2_lookup.append(lookup)

# Remap Lookup to List
f2_mapped = [[] for i in range(len(f2))]
for i, kp in enumerate(kp2_lookup):
    for j in kp.keys(): f2_mapped[kp2_lookup[i][j]].append(j)
    # for i in kp.iterkeys(): f2_mapped[kp2_lookup(i)] = i

util.plot_person(cv2.imread(im_2), f2_mapped)
'''

'''
pp = pprint.PrettyPrinter()
pp.pprint(f2_mapped)
# pp.pprint(kp2_lookup)
'''

'''
for i in range(len(pose) - 1)[:10]:
    # Load Frame Data
    frame_1 = pose_dir + '/' + pose[i]
    frame_2 = pose_dir + '/' + pose[i+1]
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

    kp2_lookup = []
    for i in range(18):
        lookup = dict()
        for j in range(len(f2)):
            # TODO: Handle case with a score threshold.
            res = kd1_frame[i].query(f2[j][i])
            lookup[tuple(f2[j][i])] = kp1_lookup[0][tuple(kd1_frame[0].data[res[1]])]
        kp2_lookup.append(lookup)

    # Remap Lookup to List
    f2_mapped = [[] for i in range(len(f2))]
    for i, kp in enumerate(kp2_lookup):
        for j in kp.keys(): f2_mapped[kp2_lookup[i][j]].append(j)
        # for i in kp.iterkeys(): f2_mapped[kp2_lookup(i)] = i

    # Update Entity Index
    print('TOTAL PERSON MAPPED: ', len(f2_mapped))
    print()
'''

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

    ''' Handle Person Count Cases '''
    if len(entities) == 0:                  # No person in previous frame (previous frame was empty)
        print('Adding New Entities...')
        # Map Pose Dictionary Indicies
        for i in range(len(frame)):
            ent_indx += 1
            entities.append(ent_indx)
            pose_map[ent_indx] = []
            print('\t> Append Entity ID: ', ent_indx)

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

    elif len(frame) == len(entities):       # Same count as previous frame
        print('No change in person count')

    elif len(frame) > len(entities):        # New person in frame
        print("Someone new came in...")

    elif len(frame) < len(entities):        # Person left frame
        print("Someone just left")

    print('Entities on Frame: ', str(entities))
    print()
