import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self,pretrained_model):
        super().__init__()
        self.model=pretrained_model
        self.classifier = nn.Softmax(dim=1)
        
    def forward(self,x):
        return self.classifier(self.model(x))
        
    