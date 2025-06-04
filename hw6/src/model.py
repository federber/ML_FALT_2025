import cv2
import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.action_space = 2
        self.discount_factor = 0.99
        self.exploration_rate = 0.1
        self.min_exploration = 0.0001
        self.total_epochs = 2000000
        self.memory_capacity = 10000
        self.batch_size = 32

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.decision_maker = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_space)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.uniform_(layer.weight, -0.01, 0.01)
                nn.init.constant_(layer.bias, 0.01)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return self.decision_maker(features)


class GameProcessor:
    @staticmethod
    def preprocess_frame(frame):
        frame = frame[0:288, 0:404]
        frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
        frame[frame > 0] = 255
        return np.expand_dims(frame, axis=2)

    @staticmethod
    def frame_to_tensor(frame):
        tensor = torch.from_numpy(frame.transpose(2, 0, 1).astype(np.float32))
        return tensor.cuda() if torch.cuda.is_available() else tensor

