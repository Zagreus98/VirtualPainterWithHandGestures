#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch.nn as nn
import torch

class PointModel(nn.Module):

    def __init__(self, num_classes=4, time_steps=16, hidden_size=64):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.LSTM = nn.LSTM(2, hidden_size, batch_first=True)
        self.linear1 = nn.Linear(hidden_size * time_steps, 10)
        self.linear2 = nn.Linear(10, num_classes)
        self.relu = nn.ReLU()
        self.time_steps = time_steps

    def forward(self, x, states):
        x = x.reshape(x.size(0), self.time_steps, 2)
        out, (h, c) = self.LSTM(x, states)  # out - (batch_size,time_steps,hidden_size) ; h - (1,batch_size,hidden_size)
        out = self.dropout(out.reshape(out.size(0), -1))
        out = self.relu(self.linear1(out))
        out = self.linear2(out)  # (batch_size,num_classes)

        return out



class PointHistoryClassifier():
    def __init__(self,model_path='model/point_history_classifier/point_history_classifier.pt',num_classes = 4):
        self.model = PointModel(num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

    def __call__(self, landmark_list):
        self.model.eval()
        landmark_list = torch.tensor(landmark_list).unsqueeze(0)
        states = (torch.randn(1, 1, 64),
                  torch.randn(1, 1, 64))
        with torch.no_grad():

            scores = self.model(landmark_list.float(),states)
            result_index = torch.argmax(scores,dim=1)

            return result_index.item()

# class PointHistoryClassifier(object):
#     def __init__(
#         self,
#         model_path='model/point_history_classifier/point_history_classifier.tflite',
#         score_th=0.5,
#         invalid_value=0,
#         num_threads=1,
#     ):
#         self.interpreter = tf.lite.Interpreter(model_path=model_path,
#                                                num_threads=num_threads)
#
#         self.interpreter.allocate_tensors()
#         self.input_details = self.interpreter.get_input_details()
#         self.output_details = self.interpreter.get_output_details()
#
#         self.score_th = score_th
#         self.invalid_value = invalid_value
#
#     def __call__(
#         self,
#         point_history,
#     ):
#         input_details_tensor_index = self.input_details[0]['index']
#         self.interpreter.set_tensor(
#             input_details_tensor_index,
#             np.array([point_history], dtype=np.float32))
#         self.interpreter.invoke()
#
#         output_details_tensor_index = self.output_details[0]['index']
#
#         result = self.interpreter.get_tensor(output_details_tensor_index)
#
#         result_index = np.argmax(np.squeeze(result))
#
#         if np.squeeze(result)[result_index] < self.score_th:
#             result_index = self.invalid_value
#
#         return result_index
