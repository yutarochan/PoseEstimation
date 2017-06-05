'''
Real-Time Pose Estimation with Person Tracking
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import os
import time
import json
from os import listdir
from os.path import isfile, join
from multiprocessing import Pool

from pose_estimate import PoseEstimation
import util
import model

model_file = '../model/pose_model.json'
model_weights = '../model/pose_model.pth'
config_file = '../model/pose_config.cfg'

'''
# Single Frame Sample
im_path = '../data/test.jpg'

start = time.time()
data = model.predict_imframe(im_path)
end = time.time()
print('TIME ELAPSED: ', end-start)

pp = pprint.PrettyPrinter()
# pp.pprint(data)
# print()

im = cv2.imread(im_path)
util.plot_person(im, data)
'''

'''
model = PoseEstimation(model_file, model_weights, config_file)

# Multiframe Test
video_folder = '../../../Dataset/PoseTest/test/'
output_folder = '../../../Dataset/PoseTest/pose_data/'

def process_frame(frame):
    print('Processing: ', frame)
    data = model.predict_imframe(video_folder + '/' + frame)
    data_dict = dict()
    data_dict['pose'] = data

    # Generate Output File
    out_name = os.path.basename(frame).split('.')[0]
    out = open(output_folder+'/'+out_name+'.json', 'wb')
    out.write(json.dumps(data_dict))
    out.close()

im_files = [f for f in listdir(video_folder) if isfile(join(video_folder, f))]
for im in im_files: process_frame(im)
'''

'''
# Unifrom Pose Plotting
video_folder = '../../../Dataset/PoseTest/test/'
plot_folder = '../../../Dataset/PoseTest/pose_data/'
pose_plot =  '../../../Dataset/PoseTest/plot/'

util.plot_sequence(video_folder, plot_folder, pose_plot)
'''

# Pose Filtering
video_source = ''
frame_folder = ''
pose_folder = ''
