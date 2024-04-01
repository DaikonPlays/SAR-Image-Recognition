import torch 
import torch.nn as nn
from model_zoo.models.utils import *

OUTPUT_CHANNELS = [256,128,256,128,256,128,512,256,512,256,512,256,512,256,1024,512,1024,512,1024,512,1024,512,1024,512,1024,512,2048,1024,2048,1024,2048,1024,2048]

class ResNext(nn.Module):
    def __init__(self, out_classes, quant_activation=False, act_bits=0, color_channels=3):
        super(ResNext, self).__init__()

        self.features = nn.Sequential(
            Convolution2D(color_channels, OUTPUT_CHANNELS[0], k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quant=quant_activation),

            Residual(OUTPUT_CHANNELS[0], OUTPUT_CHANNELS[1], OUTPUT_CHANNELS[2], quant=quant_activation),
            Residual(OUTPUT_CHANNELS[2], OUTPUT_CHANNELS[3], OUTPUT_CHANNELS[4], quant=quant_activation),
            Residual(OUTPUT_CHANNELS[4], OUTPUT_CHANNELS[5], OUTPUT_CHANNELS[6], quant=quant_activation),

            nn.MaxPool2d(kernel_size=2, stride=2),

            Residual(OUTPUT_CHANNELS[6], OUTPUT_CHANNELS[7], OUTPUT_CHANNELS[8], quant=quant_activation),
            Residual(OUTPUT_CHANNELS[8], OUTPUT_CHANNELS[9], OUTPUT_CHANNELS[10], quant=quant_activation),
            Residual(OUTPUT_CHANNELS[10], OUTPUT_CHANNELS[11], OUTPUT_CHANNELS[12], quant=quant_activation),
            Residual(OUTPUT_CHANNELS[12], OUTPUT_CHANNELS[13], OUTPUT_CHANNELS[14], quant=quant_activation),

            nn.MaxPool2d(kernel_size=2, stride=2),

            Residual(OUTPUT_CHANNELS[14], OUTPUT_CHANNELS[15], OUTPUT_CHANNELS[16], quant=quant_activation),
            Residual(OUTPUT_CHANNELS[16], OUTPUT_CHANNELS[17], OUTPUT_CHANNELS[18], quant=quant_activation),
            Residual(OUTPUT_CHANNELS[18], OUTPUT_CHANNELS[19], OUTPUT_CHANNELS[20], quant=quant_activation),
            Residual(OUTPUT_CHANNELS[20], OUTPUT_CHANNELS[21], OUTPUT_CHANNELS[22], quant=quant_activation),
            Residual(OUTPUT_CHANNELS[22], OUTPUT_CHANNELS[23], OUTPUT_CHANNELS[24], quant=quant_activation),
            Residual(OUTPUT_CHANNELS[24], OUTPUT_CHANNELS[25], OUTPUT_CHANNELS[26], quant=quant_activation),

            nn.MaxPool2d(kernel_size=2, stride=2),

            Residual(OUTPUT_CHANNELS[26], OUTPUT_CHANNELS[27], OUTPUT_CHANNELS[28], quant=quant_activation),
            Residual(OUTPUT_CHANNELS[28], OUTPUT_CHANNELS[29], OUTPUT_CHANNELS[30], quant=quant_activation),
            Residual(OUTPUT_CHANNELS[30], OUTPUT_CHANNELS[31], OUTPUT_CHANNELS[32], quant=quant_activation),

            nn.AvgPool2d(kernel_size=4, stride=1)

        )
        self.classifier = nn.Sequential( FullyConnected(OUTPUT_CHANNELS[32], out_classes, with_relu=False, quant=quant_activation) )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
