'''
Real-Time Pose Estimation with Person Tracking: Model Prediction
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import cv2
import math
import torch
import torch as T
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from torch.autograd import Variable
import scipy.ndimage.filters as filters
from scipy.ndimage.filters import gaussian_filter

import util
import config
import model as mod

torch.set_num_threads(torch.get_num_threads())

class PoseEstimation:
    def __init__(self, model_file, model_weights, config_file):
        self.param_, self.model_ = config.load_config(config_file)
        self.md, self.model = mod.init_model(model_file, model_weights)

    def predict_imframe(self, im_path):
        return self.predict_frame(cv2.imread(im_path))

    def predict_frame(self, oriImg):
        test_image = Variable(T.transpose(T.transpose(T.unsqueeze(torch.from_numpy(oriImg).float(), 0), 2, 3), 1, 2),volatile=True).cuda()
        # print('Input Image Size: ', test_image.size())

        # Multiplier: A pyramid based scaling method to evaluate image from various scales.
        multiplier = [x * self.model_['boxsize'] / oriImg.shape[0] for x in self.param_['scale_search']]
        # print('Image Scaling Multipliers: ', multiplier, '\n')

        # Heatmap and Parts Affinity Field Data Structures
        heatmap_avg = torch.zeros((len(multiplier),19,oriImg.shape[0], oriImg.shape[1])).cuda()
        paf_avg = torch.zeros((len(multiplier),38,oriImg.shape[0], oriImg.shape[1])).cuda()

        # Compute Keypoint and Part Affinity Fields
        # print('Generating Keypoint Heatmap and Parts Affinity Field Predictions...')
        for m in range(len(multiplier)):
            # Set Image Scale
            scale = multiplier[m]
            h = int(oriImg.shape[0] * scale)
            w = int(oriImg.shape[1] * scale)
            # print('[', 'Multiplier: ', scale, '-', (w, h), ']')

            # Pad Image Corresponding to Detection Stride
            pad_h = 0 if (h % self.model_['stride'] == 0) else self.model_['stride'] - (h % self.model_['stride'])
            pad_w = 0 if (w % self.model_['stride'] == 0) else self.model_['stride'] - (w % self.model_['stride'])
            new_h = h + pad_h
            new_w = w + pad_w

            # Apply Image Resize Transformation
            imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, self.model_['stride'], self.model_['padValue'])
            imageToTest_padded = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5

            # Generate Predictions
            feed = Variable(T.from_numpy(imageToTest_padded)).cuda()
            output1, output2 = self.model(feed)

            # Scale Prediction Outputs to Corresponding Image Size
            heatmap = nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])).cuda()(output2)
            paf = nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])).cuda()(output1)

            # print('Heatmap Dim:', heatmap.size())   # (1, Joint Count, X, Y)
            # print('PAF Dim:', paf.size())           # (1, PAF Count, X, Y)
            # print()

            heatmap_avg[m] = heatmap[0].data
            paf_avg[m] = paf[0].data

        # Compute Average Values
        heatmap_avg = T.transpose(T.transpose(T.squeeze(T.mean(heatmap_avg, 0)),0,1),1,2).cuda()
        paf_avg = T.transpose(T.transpose(T.squeeze(T.mean(paf_avg, 0)),0,1),1,2).cuda()

        # Convert to Numpy Type
        heatmap_avg = heatmap_avg.cpu().numpy()
        paf_avg = paf_avg.cpu().numpy()

        '''
        # [Plotting & Visualizing Heatmap and PAF]

        # Plot Heapmap Probabilities
        # util.plot_heatmap(oriImg, heatmap_avg)
        # util.plot_joint_heatmap(oriImg, heatmap_avg, 1)

        # Plot Part-Affinity Vectors
        # util.plot_paf(oriImg, paf_avg, 4)
        '''

        # Compute Heapmap Peaks (Using Non-Maximum Supression Method)
        all_peaks = []
        peak_counter = 0
        for part in range(18):
            # Smooth out heapmap with gaussian kernel to remove high frequency variation.
            map_ori = heatmap_avg[:,:,part]
            map = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map.shape)
            map_left[1:,:] = map[:-1,:]
            map_right = np.zeros(map.shape)
            map_right[:-1,:] = map[1:,:]
            map_up = np.zeros(map.shape)
            map_up[:,1:] = map[:,:-1]
            map_down = np.zeros(map.shape)
            map_down[:,:-1] = map[:,1:]

            # Compute Peak Based on Binary Threshold
            peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > self.param_['thre1']))
            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse

            # Derive Joint Keypoint Peaks with Mapped ID with Probabilities
            peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        # TODO: Output Predictions
        # TODO: Plot Keypoint Predictions (Points with Color as Probabilities)
        # print('Peak Count: ', peak_counter)

        # print(peaks_with_score) # Return as one of the parameters
        # !!RETURN_ME!!

        '''
        # [Plot KeyPoint (with Probabilities)]
        # util.plot_key_point(oriImg, all_peaks)
        '''

        # Load Joint Index and Sequences Data
        mapIdx = self.md.get_mapIdx()
        limbSeq = self.md.get_limbseq()

        '''
        # Find Parts Connection and Cluster to Different Subsets
        subsets = []
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        connection = dict()

        for k in range(len(mapIdx))[:1]:
            print('='*80)
            print('MAP INDEX: ', mapIdx[k], '\n')

            score_mid = paf_avg[:,:,[x-19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0]-1]
            candB = all_peaks[limbSeq[k][1]-1]

            print('CANDIDATE A: ', str(candA))
            print('CANDIDATE B: ', str(candB))
            print()

            connection[k] = []
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]

            # Add Parts to Subset in Special Cases
            if nA == 0 and nB == 0: continue
            elif nA == 0:
                print('Handle Special Case')
            elif nB == 0:
                print('Handle Special Case')

            temp = []
        '''

        # Compute Part-Affinity Fields
        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(mapIdx)):
            score_mid = paf_avg[:,:,[x-19 for x in mapIdx[k]]]
            # print(score_mid.shape)

            candA = all_peaks[limbSeq[k][0]-1]
            candB = all_peaks[limbSeq[k][1]-1]
            # print('Limb Seq Connection: [', limbSeq[k][0]-1, ',', limbSeq[k][1]-1, ']\n')

            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]

            if nA != 0 and nB != 0:
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):

                        # Compute Joint Unit Vector
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                        # Assert: Check if the norm is a not a zero vector.
                        if not np.any(norm):
                            #print('Exception: Norm is a zero-vector')
                            continue

                        # TODO: Save this vector!
                        vec = np.divide(vec, norm)
                        #print('Unit Vector: [',i, ', ', j, ']: ', str(vec))

                        startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), np.linspace(candA[i][1], candB[j][1], num=mid_num))
                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in range(len(startend))])

                        # Compute Components for Affinity Field Criterion
                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)

                        # Check PAF Criterion
                        criterion1 = len(np.nonzero(score_midpts > self.param_['thre2'])[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0,5))

                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if (i not in connection[:,3] and j not in connection[:,4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if len(connection) >= min(nA, nB): break

                connection_all.append(connection)

                #print('\nConnections:')
                #print(connection)
                #print()
            else:
                # Handle Exception for Potential Missing Part Entities
                special_k.append(k)
                connection_all.append([])

        # TODO: Create a data structure to hold all of the PAF midpoint for each joint connection based on the above derivation.
        # Use peak finding algorithm again to get the peaks of the PAF and derive a PA Vector
        # !!RETURN_ME!!

        # Build Human Pose
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:,0]
                partBs = connection_all[k][:,1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])):
                    found = 0
                    subset_idx = [-1, -1]

                    for j in range(len(subset)):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2: # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        # print "found = 2"
                        membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0: #merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else: # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        # Remove Rows of Subset with the Least Parts Available
        deleteIdx = [];
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

        # print('TOTAL PEOPLE DETECTED: ', str(len(subset)))

        # Setup Data Structure for Return
        # Data Structure: (person, limb seq index, x, y)
        limb_midpts = []
        for n in range(len(subset)):
            for i in range(17):
                index = subset[n][np.array(limbSeq[i])-1]
                if -1 in index: continue

                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]

                mX = np.mean(X)
                mY = np.mean(Y)

                limb_midpts.append((n, i, mX, mY))

        pose_data = dict()
        # for i in range(subset(i)):


        return all_peaks, limb_midpts
