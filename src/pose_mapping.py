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
import operator
import itertools
from os import listdir
from scipy.spatial import KDTree
from os.path import isfile, join

import util

# Multiframe Mapping
pose_dir = '../data/pose_data/'
image_dir = '../data/test/'
pose = sorted([f for f in listdir(pose_dir) if isfile(join(pose_dir, f)) and f.split('.')[-1] == 'json'])

pose_map = dict()   # Dictionary to keep track of individual entities in frame.
entities = []       # Index frame in dictionary to track entity id.
ent_indx = -1       # Last added entity index.

# Following to Keep Track from Previous Frame
kd_frame  = []      # KD-Tree Pose Tree List
kp_lookup = []      # Key Point Lookup Dictionary List

for i in range(len(pose))[:16]:
    print('FRAME: ', i)
    # Load Frame Data
    frame = json.loads(open(pose_dir+'/'+pose[i], 'rb').read())['pose']

    # Handle Entity Tracking Cases
    if len(entities) == 0:                  # No person in previous frame (previous frame was empty)
        print('Adding New Entities...')

        # Map Pose Dictionary Indicies
        for x in range(len(frame)):
            ent_indx += 1
            entities.append(ent_indx)
            pose_map[ent_indx] = []
            pose_map[ent_indx].append([tuple(x) for x in frame[x]])
            print('\t> Append Entity ID: ', ent_indx)

    elif len(frame) == len(entities):       # Same count as previous frame
        print('No Changes in Entity Count...')

        # Build Lookup Mapping for Current Frame
        curr_lookup = []
        for x in range(18):
            lookup = dict()
            cand = []
            for y in range(len(frame)):
                # FIXME: Figure out joint conflict management tactic for duplicate point id.
                # TODO: Test joint conflict management in all point cases.
                # res = kd_frame[x].query(frame[y][x])

                query = kd_frame[x].query(frame[y][x], len(frame))
                cand.append(zip(query[0], query[1]))

                # lookup[tuple(frame[y][x])] = kp_lookup[0][tuple(kd_frame[0].data[res[1]])]

            # Compute Candidate Point Maps
            cand = [(c[1], (c[0], _id)) for _id, can in enumerate(cand) for c in can]     # Flatten List with Candidate ID Added
            cand = [(k, map(operator.itemgetter(1), g)) for k, g in itertools.groupby(sorted(cand), key=operator.itemgetter(0))]
            final = [min(c[1], key=operator.itemgetter(0)) for c in cand]
            print(final)

            for idx, r in enumerate(final):
                lookup[tuple(frame[idx][x])] = kp_lookup[0][tuple(kd_frame[0].data[r[1]])]
            # print([k for k, sub in itertools.groupby(cand, operator.itemgetter(0))])

            curr_lookup.append(lookup)

        # Remap Lookup to List
        mapped = [[] for i in range(len(frame))]
        for x, kp in enumerate(curr_lookup):
            for y in kp.keys(): mapped[curr_lookup[x][y]].append(y)

        for x in range(len(mapped)):
            print('JOINT COUNT FOR ', x, ': ', len(mapped[x]))

        # Append Mapped Entirs to Pose Map Dictionary
        for i, p in enumerate(mapped): pose_map[i].append(p)

        util.plot_person(cv2.imread(image_dir + '/' + pose[i].split('.')[0] + '.jpg'), mapped)

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
