import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import math
import cv2

class ConvNet(nn.Module):
    def __init__(self, input_dims, numOutputs, chckptDir):
        super().__init__()
        self.input_dims = input_dims
        self.chckptDir = chckptDir
        self.conv = nn.Sequential(
            nn.Conv2d(input_dims[0], 32, 8, stride=2),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 4, stride=1),
            nn.MaxPool2d(2,2),
            nn.Flatten()
        )

        fc_input_dims = self.find_input_dims()
        self.fc = nn.Linear(fc_input_dims, 128)
        self.nc1 = nn.Linear(128,256)
        self.nc2 = nn.Linear(256,numOutputs)

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def find_input_dims(self):
        inputs = torch.zeros(self.input_dims).unsqueeze(0)
        x = self.conv(inputs)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = torch.tensor(x).to(torch.float32).to(self.device)
        print(x.shape)
        #print(x.shape)
        x = self.conv(x)
        #print(x.shape)
        x = F.relu(self.fc(x))
        x = F.relu(self.nc1(x))
        x = F.softmax(self.nc2(x))
        
        print(x.shape)

        return x

    def save_model(self):
        torch.save(self.state_dict(), self.chckptDir)

    def load_model(self):
        self.load_state_dict(torch.load(self.chckptDir))


