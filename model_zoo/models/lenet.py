import torch
import torch.nn as nn
from model_zoo.models.utils import * 

class LeNet(nn.Module):
    def __init__(self, input_channels, out_classes, quantization=False, act_int=0, act_dec=0, e2cnn_size=1 ):
        super(LeNet, self).__init__()
        if e2cnn_size == 1:
            structure = [6,16,120,84]
        elif e2cnn_size == 2:
            structure = [6,16,120,84]
        self.features = nn.Sequential(
                Convolution2D(input_channels, structure[0], k_size=5, stride=1, padding=0, with_bn=False, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
                nn.MaxPool2d(2),
                Convolution2D(structure[0], structure[1], k_size=5, stride=1, padding=0, with_bn=False, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
                nn.MaxPool2d(2), 
                Convolution2D(structure[1], structure[2], k_size=5, stride=1, padding=0, with_bn=False, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec)
                )


        self.classifier = nn.Sequential(
            FullyConnected(structure[2], structure[3], with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            FullyConnected(structure[3], out_classes, with_relu=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec)
            )
        

    def forward(self, x):
        x = self.features(x)
        x = torch.reshape(x, (x.shape[0], -1))
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x