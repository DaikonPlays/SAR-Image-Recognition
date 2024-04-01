import torch 
import torch.nn as nn
from model_zoo.models.utils import *

#AS DEFINED HERE FOR CIFAR10: https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-alexnet-cifar10.ipynb
class AlexNet(nn.Module):
    def __init__(self, input_channels, out_classes, quantization=False, act_int=0, act_dec=0, e2cnn_size=1 ):
        super(AlexNet, self).__init__()
        if e2cnn_size == 1:
            structure = [64,192,384,256,256,4096,4096]
        elif e2cnn_size == 2:
            structure = [64,192,384,256,256,4096,4096]
        
        self.features = nn.Sequential(
            Convolution2D(input_channels, structure[0], k_size=5, stride=1, padding=0, with_bn=False, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            Convolution2D(structure[0], structure[1], k_size=5, stride=2, padding=1, with_bn=False, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            Convolution2D(structure[1], structure[2], k_size=3, stride=1, padding=1, with_bn=False, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            Convolution2D(structure[2], structure[3], k_size=3, stride=1, padding=1, with_bn=False, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            Convolution2D(structure[3], structure[4], k_size=3, stride=1, padding=1, with_bn=False, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0)

        )
        self.classifier = nn.Sequential(
            FullyConnected(structure[4], structure[5], with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            FullyConnected(structure[5], structure[6], with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            FullyConnected(structure[6], out_classes, with_relu=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec)      
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
