import torch
import torch.nn as nn
from model_zoo.models.utils import *


class VGG16(nn.Module):
    def __init__(self, input_channels, out_classes, quantization=False, act_int=0, act_dec=0, e2cnn_size=1 ):
        super(VGG16, self).__init__()
        if e2cnn_size == 1:
            structure = [64,64,128,128,256,256,256,512,512,512,512,512,512,4096, 4096]
        elif e2cnn_size == 2:
            structure = [64,64,128,128,256,256,256,512,512,512,512,512,512,4096, 4096]

        self.features = nn.Sequential(
            Convolution2D(input_channels, structure[0], k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            Convolution2D(structure[0], structure[1], k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            Convolution2D(structure[1], structure[2], k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            Convolution2D(structure[2], structure[3], k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            Convolution2D(structure[3], structure[4], k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            Convolution2D(structure[4], structure[5], k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            Convolution2D(structure[5], structure[6], k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            Convolution2D(structure[6], structure[7], k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            Convolution2D(structure[7], structure[8], k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            Convolution2D(structure[8], structure[9], k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            Convolution2D(structure[9], structure[10], k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            Convolution2D(structure[10], structure[11], k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            Convolution2D(structure[11], structure[12], k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),

            nn.AvgPool2d(kernel_size=2, stride=2, padding=0) 
            )

        self.classifier = nn.Sequential(
            FullyConnected(structure[12], structure[13], with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            FullyConnected(structure[13], structure[14], with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            FullyConnected(structure[14], out_classes, with_relu=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec)
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x