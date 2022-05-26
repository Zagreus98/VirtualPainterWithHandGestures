#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class KeyPointClassifier():
    def __init__(self,config,num_classes = 8):

        self.model = nn.Sequential(
                        nn.Linear(42,20),
                        nn.ReLU(),
                        nn.Dropout(p=0.4),
                        nn.Linear(20,10),
                        nn.ReLU(),
                        nn.Linear(10,num_classes)
        )
        self.model.load_state_dict(torch.load(config.gesture_classification.model_path,map_location=torch.device('cpu')))
        self.score_th = config.gesture_classification.score_th

    def __call__(self, landmark_list):
        self.model.eval()
        landmark_list = torch.tensor(landmark_list).unsqueeze(0)
        with torch.no_grad():
            scores = self.model(landmark_list.float())
            scores = F.softmax(scores,dim=1)
            result_index = torch.argmax(scores,dim=1)

            if scores.squeeze(0)[result_index] < self.score_th:
                return 7
            else:
                return result_index.item()

