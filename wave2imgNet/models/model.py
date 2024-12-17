import torch
from torch import nn
import torch.nn.functional as F
import timm
from .layers import LambdaConv

model_name = "efficientnetv2_rw_s"
class TimmModel(nn.Module):
    def __init__(self, outdim, lambda_nn=True):
        super(TimmModel, self).__init__()
        self.outdim = outdim
        self.lambda_nn = lambda_nn
        self.lambda_heads=4
        self.backbone = timm.create_model(model_name, pretrained=False)
        self.replace()


    def replace(self):
        if self.lambda_nn:
            num_feature=self.backbone.conv_head.out_channels
            self.backbone.bn2 = nn.Sequential(
                    LambdaConv(num_feature, num_feature, heads=self.lambda_heads, k=16, u=1),
                    nn.BatchNorm2d(num_feature),
                    nn.SiLU(inplace=True)
                    )
        self.backbone.classifier = nn.Linear(self.backbone.conv_head.out_channels, 2)

    def forward(self, x):
        x = self.backbone(x) 
        return x

