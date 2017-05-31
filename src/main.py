'''
Real-Time Pose Estimation with Person Tracking
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import os
import cv2
import time
import pprint

from pose_estimate import PoseEstimation
import util
import model

model_file = '../model/pose_model.json'
model_weights = '../model/pose_model.pth'
config_file = '../model/pose_config.cfg'

# Single Frame Sample
im_path = '../data/test2.jpg'

model = PoseEstimation(model_file, model_weights, config_file)

start = time.time()
data = model.predict_imframe(im_path)
end = time.time()
print('TIME ELAPSED: ', end-start)

pp = pprint.PrettyPrinter()
pp.pprint(data)
print()

# im = cv2.imread(im_path)
# util.plot_person(im, data)
