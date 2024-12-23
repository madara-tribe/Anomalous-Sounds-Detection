from torch import nn
import timm
from .layers import LambdaConv

class WaveVisonNet(nn.Module):
    def __init__(self, output_dim, use_lambda_layers=True, backbone_name="efficientnetv2_rw_s", lambda_heads=4):
        super(WaveVisonNet, self).__init__()
        self.output_dim = output_dim
        self.use_lambda_layers = use_lambda_layers
        self.lambda_heads = lambda_heads
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(backbone_name, pretrained=False)
        self._customize_backbone()


    def _customize_backbone(self):
        if self.use_lambda_layers:
            num_feature=self.backbone.conv_head.out_channels
            self.backbone.bn2 = nn.Sequential(
                    LambdaConv(num_feature, num_feature, heads=self.lambda_heads, k=16, u=1),
                    nn.BatchNorm2d(num_feature),
                    nn.SiLU(inplace=True)
                    )
        self.backbone.classifier = nn.Linear(self.backbone.conv_head.out_channels, self.output_dim)

    def forward(self, x):
        x = self.backbone(x) 
        return x

