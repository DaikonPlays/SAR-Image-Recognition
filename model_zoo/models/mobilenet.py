import torch 
import torch.nn as nn
from model_zoo.models.utils import *

OUTPUT_CHANNELS = [32,64,128,128,256,256,512,512,512,512,512,512,1024,1024]


class MobileNet(nn.Module):
    def __init__(self, out_classes, quant_activation=False, act_bits=0, color_channels=3):
        super(MobileNet, self).__init__()

        self.features = nn.Sequential(
            Convolution2D(color_channels, OUTPUT_CHANNELS[0], k_size=3, stride=1, padding=0, with_bn=True, with_relu=True, quant=quant_activation),
            SeparableConvolution2D(OUTPUT_CHANNELS[0], OUTPUT_CHANNELS[1], k_size=3, stride=1, padding=0, with_bn=True, with_relu=True, quant=quant_activation),
            SeparableConvolution2D(OUTPUT_CHANNELS[1], OUTPUT_CHANNELS[2], k_size=3, stride=1, padding=0, with_bn=True, with_relu=True, quant=quant_activation),
            SeparableConvolution2D(OUTPUT_CHANNELS[2], OUTPUT_CHANNELS[3], k_size=3, stride=1, padding=0, with_bn=True, with_relu=True, quant=quant_activation),
            SeparableConvolution2D(OUTPUT_CHANNELS[3], OUTPUT_CHANNELS[4], k_size=3, stride=1, padding=0, with_bn=True, with_relu=True, quant=quant_activation),
            SeparableConvolution2D(OUTPUT_CHANNELS[4], OUTPUT_CHANNELS[5], k_size=3, stride=1, padding=0, with_bn=True, with_relu=True, quant=quant_activation),
            SeparableConvolution2D(OUTPUT_CHANNELS[5], OUTPUT_CHANNELS[6], k_size=3, stride=1, padding=0, with_bn=True, with_relu=True, quant=quant_activation),
            SeparableConvolution2D(OUTPUT_CHANNELS[6], OUTPUT_CHANNELS[7], k_size=3, stride=1, padding=0, with_bn=True, with_relu=True, quant=quant_activation),
            SeparableConvolution2D(OUTPUT_CHANNELS[7], OUTPUT_CHANNELS[8], k_size=3, stride=1, padding=0, with_bn=True, with_relu=True, quant=quant_activation),
            SeparableConvolution2D(OUTPUT_CHANNELS[8], OUTPUT_CHANNELS[9], k_size=3, stride=1, padding=0, with_bn=True, with_relu=True, quant=quant_activation),
            SeparableConvolution2D(OUTPUT_CHANNELS[9], OUTPUT_CHANNELS[10], k_size=3, stride=1, padding=0, with_bn=True, with_relu=True, quant=quant_activation),
            SeparableConvolution2D(OUTPUT_CHANNELS[10], OUTPUT_CHANNELS[11], k_size=3, stride=1, padding=0, with_bn=True, with_relu=True, quant=quant_activation),
            SeparableConvolution2D(OUTPUT_CHANNELS[11], OUTPUT_CHANNELS[12], k_size=3, stride=1, padding=0, with_bn=True, with_relu=True, quant=quant_activation),
            SeparableConvolution2D(OUTPUT_CHANNELS[12], OUTPUT_CHANNELS[13], k_size=3, stride=1, padding=0, with_bn=True, with_relu=True, quant=quant_activation),

            nn.AvgPool2d(kernel_size=4, stride=1)
        )
        self.classifier = nn.Sequential( FullyConnected(OUTPUT_CHANNELS[13], out_classes, with_relu=False, quant=quant_activation) )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x