import torch 
import torch.nn as nn
from model_zoo.models.utils import *

class MobileNetV2(nn.Module):
    def __init__(self, input_channels, out_classes, quantization=False, act_int=0, act_dec=0, e2cnn_size=1 ):
        super(MobileNetV2, self).__init__()
        if e2cnn_size == 1:
            structure = [32, 16, 24, 32, 64, 96, 160, 320, 1280]
        elif e2cnn_size == 2:
            structure = [32, 16, 24, 32, 64, 96, 160, 320, 1280]

        self.features = nn.Sequential(
            Convolution2D(input_channels, structure[0], k_size=3, stride=2, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),

            BottleneckResidual(structure[0], 1, structure[1], reduce_dim=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),

            BottleneckResidual(structure[1], 6, structure[2], reduce_dim=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            BottleneckResidual(structure[2], 6, structure[2], reduce_dim=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec),

            BottleneckResidual(structure[2], 6, structure[3], reduce_dim=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            BottleneckResidual(structure[3], 6, structure[3], reduce_dim=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            BottleneckResidual(structure[3], 6, structure[3], reduce_dim=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec),

            BottleneckResidual(structure[3], 6, structure[4], reduce_dim=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            BottleneckResidual(structure[4], 6, structure[4], reduce_dim=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            BottleneckResidual(structure[4], 6, structure[4], reduce_dim=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            BottleneckResidual(structure[4], 6, structure[4], reduce_dim=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec),

            BottleneckResidual(structure[4], 6, structure[5], reduce_dim=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            BottleneckResidual(structure[5], 6, structure[5], reduce_dim=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            BottleneckResidual(structure[5], 6, structure[5], reduce_dim=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec),

            BottleneckResidual(structure[5], 6, structure[6], reduce_dim=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            BottleneckResidual(structure[6], 6, structure[6], reduce_dim=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            BottleneckResidual(structure[6], 6, structure[6], reduce_dim=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec),

            BottleneckResidual(structure[6], 6, structure[7], reduce_dim=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),

            Convolution2D(structure[7], structure[8], k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),

            nn.AvgPool2d(kernel_size=2, stride=1)
        )
        self.classifier = FullyConnected(structure[8], out_classes, with_relu=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec) 

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x