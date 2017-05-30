'''
Real-Time Pose Estimation with Person Tracking
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import os
import pprint

from pose_estimate import PoseEstimation
import model

model_file = '../model/pose_model.json'
model_weights = '../model/pose_model.pth'
config_file = '../model/pose_config.cfg'

im_path = '../data/test.jpg'

model = PoseEstimation(model_file, model_weights, config_file)
keypoint, midpoint = model.predict_imframe(im_path)

'''
pp = pprint.PrettyPrinter()
pp.pprint(keypoint[0])
print()

pp.pprint(midpoint)
'''
