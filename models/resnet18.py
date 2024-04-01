import torch 
import torch.nn as nn
from model_zoo.models.utils import *


class ResNet18(nn.Module):
    def __init__(self, input_channels, out_classes, quantization=False, act_int=0, act_dec=0, e2cnn_size=1 ):
        super(ResNet18, self).__init__()
        if e2cnn_size == 1:
            structure = [16, 16, 16, 32, 32, 32, 64, 64, 64, 64]
        elif e2cnn_size == 2:
            structure = [16, 16, 16, 16, 32, 32, 32, 64, 64, 64]

        self.conv = Convolution2D(input_channels, structure[0], k_size=3, stride=1, padding=1, with_bn=False, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec)

        self.residual1 = nn.Sequential( 
            ResidualLayer(structure[0], structure[1], skip_proj=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            ResidualLayer(structure[1], structure[2], skip_proj=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec), 
            ResidualLayer(structure[2], structure[3], skip_proj=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec) 
            )

        self.residual2 = nn.Sequential( 
            ResidualLayer(structure[3], structure[4], skip_proj=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            ResidualLayer(structure[4], structure[5], skip_proj=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec), 
            ResidualLayer(structure[5], structure[6], skip_proj=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec) 
            )

        self.residual3 = nn.Sequential( 
            ResidualLayer(structure[6], structure[7], skip_proj=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            ResidualLayer(structure[7], structure[8], skip_proj=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec), 
            ResidualLayer(structure[8], structure[9], skip_proj=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec) 
            )


        self.pooling = nn.AvgPool2d(kernel_size=4, stride=1, padding=0) 

        self.classifier = FullyConnected(structure[9], out_classes, with_relu=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec)

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = self.residual1(x)
        # print(x.shape)
        x = self.residual2(x)
        # print(x.shape)
        x = self.residual3(x)
        # print(x.shape)
        # input()
        
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # print(x.shape)
        return x