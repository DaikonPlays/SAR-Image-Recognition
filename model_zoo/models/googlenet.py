import torch 
import torch.nn as nn
from model_zoo.models.utils import *

class GoogLeNet(nn.Module):
    def __init__(self, input_channels, out_classes, quantization=False, act_int=0, act_dec=0, e2cnn_size=1 ):
        super(GoogLeNet, self).__init__()  
        if e2cnn_size == 1:
            structure = [64,64,192, 64,96,128,16,32,32, 128,128,192,32,96,64, 192,96,208,16,48,64, 160,112,224,24,64,64, 128,128,256,24,64,64, 112,144,288,32,64,64, 256,160,320,32,128,128,
                         256,160,320,32,128,128, 384,192,384,48,128,128]
        elif e2cnn_size == 2:
            structure = [64,64,192,64,96,128,16,32,32,128,128,192,32,96,64,192,96,208,16,48,64,160,112,224,24,64,64,128,128,256,24,64,64,112,144,288,32,64,64,256,160,320,32,128,128,256,160,320,32,128,128,384,192,384,48,128,128]


        IN1_IN = structure[2]
        IN2_IN = structure[3] + structure[5] + structure[7] + structure[8]
        IN3_IN = structure[9] + structure[11] + structure[13] + structure[14]
        IN4_IN = structure[15] + structure[17] + structure[19] + structure[20]
        IN5_IN = structure[21] + structure[23] + structure[25] + structure[26]
        IN6_IN = structure[27] + structure[29] + structure[31] + structure[32]
        IN7_IN = structure[33] + structure[35] + structure[37] + structure[38]
        IN8_IN = structure[39] + structure[41] + structure[43] + structure[44]
        IN9_IN = structure[45] + structure[47] + structure[49] + structure[50]
        FC_IN  = structure[51] + structure[53] + structure[55] + structure[56]

        self.features = nn.Sequential(
            Convolution2D(input_channels, structure[0], k_size=3, stride=2, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),

            nn.MaxPool2d(kernel_size=2, stride=2),

            Convolution2D(structure[0], structure[1], k_size=1, stride=1, padding=0, with_bn=True, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),
            Convolution2D(structure[1], structure[2], k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=act_int, dec_bits=act_dec),

            # nn.MaxPool2d(kernel_size=2, stride=2),

            Inception(IN1_IN, structure[3], structure[4], structure[5], structure[6], structure[7], structure[8], quantization=False, int_bits=0, dec_bits=0),
            Inception(IN2_IN, structure[9], structure[10], structure[11], structure[12], structure[13], structure[14], quantization=False, int_bits=0, dec_bits=0),

            nn.MaxPool2d(kernel_size=2, stride=2),

            Inception(IN3_IN, structure[15], structure[16], structure[17], structure[18], structure[19], structure[20], quantization=False, int_bits=0, dec_bits=0),
            Inception(IN4_IN, structure[21], structure[22], structure[23], structure[24], structure[25], structure[26], quantization=False, int_bits=0, dec_bits=0),
            Inception(IN5_IN, structure[27], structure[28], structure[29], structure[30], structure[31], structure[32], quantization=False, int_bits=0, dec_bits=0),
            Inception(IN6_IN, structure[33], structure[34], structure[35], structure[36], structure[37], structure[38], quantization=False, int_bits=0, dec_bits=0),
            Inception(IN7_IN, structure[39], structure[40], structure[41], structure[42], structure[43], structure[44], quantization=False, int_bits=0, dec_bits=0),

            nn.MaxPool2d(kernel_size=2, stride=2),

            Inception(IN8_IN, structure[45], structure[46], structure[47], structure[48], structure[49], structure[50], quantization=False, int_bits=0, dec_bits=0),
            Inception(IN9_IN, structure[51], structure[52], structure[53], structure[54], structure[55], structure[56], quantization=False, int_bits=0, dec_bits=0),

            nn.AvgPool2d(kernel_size=2, stride=1)
        )
        self.classifier = FullyConnected(FC_IN, out_classes, with_relu=False, quantization=quantization, int_bits=act_int, dec_bits=act_dec) 

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
