#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch.nn as nn
import torch

class KeyPointClassifier():
    def __init__(self,model_path='model/keypoint_classifier/keypoint_classifier.pt',num_classes = 4):
        self.model = nn.Sequential(
                        nn.Linear(42,20),
                        nn.ReLU(),
                        nn.Dropout(p=0.4),
                        nn.Linear(20,10),
                        nn.ReLU(),
                        nn.Linear(10,num_classes)
        )
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

    def __call__(self, landmark_list):
        self.model.eval()
        landmark_list = torch.tensor(landmark_list).unsqueeze(0)
        with torch.no_grad():
            scores = self.model(landmark_list.float())
            result_index = torch.argmax(scores,dim=1)

            return result_index.item()

