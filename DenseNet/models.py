import torch.nn as nn
from modules import densenet121


class ResNetModel(nn.Module):
    def __init__(self, weights: str):
        super(ResNetModel, self).__init__()
        self.model = densenet121(weights=weights)
        self.model.fc = nn.Linear(512, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, gt=None):
        output = self.model(x)

        if gt != None:
            loss = self.criterion(output, gt)
            return output, loss

        return output
