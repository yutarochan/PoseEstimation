'''
Real-Time Pose Estimation: Model Utility Functions
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import os
import json
import time
import torch
import torch.nn as nn

class ModelData:
    def __init__(self, model_path):
        self.model = json.loads(open(model_path, 'rb').read())

    def load_model(self):
        layers = []

        # Phase 1 Model (VGG-19 Feature Extraction Layers)
        for i in range(len(self.model['blocks']['block0'])):
            one_ = self.model['blocks']['block0'][i]
            for k,v in one_.iteritems():
                if 'pool' in k:
                    layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
                else:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
                    layers += [conv2d, nn.ReLU(inplace=True)]

        # Construct Rest of Model
        models = {}
        models['block0']=nn.Sequential(*layers)

        blocks = dict(self.model['blocks'])
        blocks.pop('block0', None)
        for k,v in blocks.iteritems():
            models[k] = self.make_layers(v)

        return models

    def make_layers(self, cfg_dict):
        layers = []
        for i in range(len(cfg_dict)-1):
            one_ = cfg_dict[i]
            for k,v in one_.iteritems():
                if 'pool' in k:
                    layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
                else:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
                    layers += [conv2d, nn.ReLU(inplace=True)]
        one_ = cfg_dict[-1].keys()
        k = one_[0]
        v = cfg_dict[-1][k]
        conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
        layers += [conv2d]
        return nn.Sequential(*layers)

    def get_limbseq(self):
        return self.model['limbSeq']

    def get_mapIdx(self):
        return self.model['mapIdx']

    def get_colors(self):
        return self.model['colors']

class PoseModel(nn.Module):
    def __init__(self,model_dict,transform_input=False):
        super(PoseModel, self).__init__()
        self.model0   = model_dict['block0']
        self.model1_1 = model_dict['block1_1']
        self.model2_1 = model_dict['block2_1']
        self.model3_1 = model_dict['block3_1']
        self.model4_1 = model_dict['block4_1']
        self.model5_1 = model_dict['block5_1']
        self.model6_1 = model_dict['block6_1']

        self.model1_2 = model_dict['block1_2']
        self.model2_2 = model_dict['block2_2']
        self.model3_2 = model_dict['block3_2']
        self.model4_2 = model_dict['block4_2']
        self.model5_2 = model_dict['block5_2']
        self.model6_2 = model_dict['block6_2']

    def forward(self, x):
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2  = torch.cat([out1_1,out1_2,out1],1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3   = torch.cat([out2_1,out2_2,out1],1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4   = torch.cat([out3_1,out3_2,out1],1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5   = torch.cat([out4_1,out4_2,out1],1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6   = torch.cat([out5_1,out5_2,out1],1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return out6_1, out6_2

def init_model(model_file, model_weights):
    md = ModelData(model_file)

    model = PoseModel(md.load_model())
    model.load_state_dict(torch.load(model_weights))
    model.cuda().float().eval()

    return md, model

if __name__ == '__main__':
    ''' Unit Testing '''
    model_file = '../model/pose_model.json'
    model_weights = '../model/pose_model.pth'

    md, model = init_model(model_file, model_weights)

    print('Limb Sequence:')
    print(str(md.get_limbseq()) + '\n')

    print('Map Index:')
    print(str(md.get_mapIdx()) + '\n')

    print('Colors')
    print(str(md.get_colors()) + '\n')

    # Test Load Model
    print('Model:')
    print(model)
