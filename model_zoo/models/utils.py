import torch
import torch.nn as nn

class QuantizeActivation(nn.Module):
    def __init__(self, int_bits=0, dec_bits=0, quantization=False):
        super(QuantizeActivation, self).__init__()
        self.int_bits = int_bits
        self.dec_bits = dec_bits
        self.quantization = quantization
        if(quantization):
            self.scale = (2 ** -self.dec_bits)
            self.zero_point = 0
            self.qmin = -2 ** (self.int_bits + self.dec_bits)
            self.qmax = 2 ** (self.int_bits + self.dec_bits) - 1

        
    def get_int_bits(self):
        return self.int_bits
    def get_dec_bits(self):
        return self.dec_bits
    def set_int_bits(self, n):
        self.int_bits = n
        self.qmin = -2 ** (self.int_bits + self.dec_bits)
        self.qmax = 2 ** (self.int_bits + self.dec_bits) - 1
    def set_dec_bits(self, n):
        self.dec_bits = n
        self.scale = (2 ** -self.dec_bits)
        self.qmin = -2 ** (self.int_bits + self.dec_bits)
        self.qmax = 2 ** (self.int_bits + self.dec_bits) - 1

    def set_quantization(self):
        self.quantization = True
    def unset_quantization(self):
        self.quantization = False

    def forward(self, x):
        if(self.quantization):
            return torch.fake_quantize_per_tensor_affine(x, scale=self.scale, zero_point=self.zero_point, quant_min=self.qmin, quant_max=self.qmax)
        else:
            return x


class Convolution2D(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride=1, padding=0, bias=True, dilation=1, with_bn=False, with_relu=False, quantization=False, int_bits=0, dec_bits=0):
        super(Convolution2D, self).__init__()
        convolution = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=padding, stride=stride, bias=bias, dilation=dilation)
        quantization = QuantizeActivation(int_bits=int_bits, dec_bits=dec_bits, quantization=quantization)
        if with_bn:
            if with_relu:
                self.operation = nn.Sequential(quantization, convolution, quantization, nn.BatchNorm2d(out_channels), nn.ReLU())
            else:
                self.operation = nn.Sequential(quantization, convolution, quantization, nn.BatchNorm2d(out_channels))
        else:
            if with_relu:
                self.operation = nn.Sequential(quantization, convolution, quantization, nn.ReLU())
            else:
                self.operation = nn.Sequential(quantization, convolution, quantization)

    def forward(self, inputs):
        # print(" CONV INPUT: {}".format(inputs.shape))
        outputs = self.operation(inputs)
        # print(" CONV OUTPUT: {}".format(outputs.shape))
        return outputs

class DepthwiseConvolution2D(nn.Module):
    def __init__(self, channels, k_size, stride=1, padding=0, bias=True, dilation=1, with_bn=False, with_relu=False, quantization=False, int_bits=0, dec_bits=0):
        super(DepthwiseConvolution2D, self).__init__()
        convolution = nn.Conv2d(channels, channels, groups=channels, kernel_size=k_size, padding=padding, stride=stride, bias=bias, dilation=dilation)
        quantization = QuantizeActivation(int_bits=int_bits, dec_bits=dec_bits, quantization=quantization)
        if with_bn:
            if with_relu:
                self.operation = nn.Sequential(quantization, convolution, quantization, nn.BatchNorm2d(channels), nn.ReLU())
            else:
                self.operation = nn.Sequential(quantization, convolution, quantization, nn.BatchNorm2d(channels))
        else:
            if with_relu:
                self.operation = nn.Sequential(quantization, convolution, quantization, nn.ReLU())
            else:
                self.operation = nn.Sequential(quantization, convolution, quantization)

    def forward(self, inputs):
        outputs = self.operation(inputs)
        return outputs

class PointwiseConvolution2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=0, bias=True, dilation=1, with_bn=False, with_relu=False, quantization=False, int_bits=0, dec_bits=0):
        super(PointwiseConvolution2D, self).__init__()
        convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=padding, stride=stride, bias=bias, dilation=dilation)
        quantization = QuantizeActivation(int_bits=int_bits, dec_bits=dec_bits, quantization=quantization)
        if with_bn:
            if with_relu:
                self.operation = nn.Sequential(quantization, convolution, quantization, nn.BatchNorm2d(out_channels), nn.ReLU())
            else:
                self.operation = nn.Sequential(quantization, convolution, quantization, nn.BatchNorm2d(out_channels))
        else:
            if with_relu:
                self.operation = nn.Sequential(quantization, convolution, quantization, nn.ReLU())
            else:
                self.operation = nn.Sequential(quantization, convolution, quantization)

    def forward(self, inputs):
        outputs = self.operation(inputs)
        return outputs



class BottleneckResidual(nn.Module):
    def __init__(self, ch_in, expansion, ch_out, reduce_dim, quantization=False, int_bits=0, dec_bits=0):
        super(BottleneckResidual, self).__init__()
        self.reduce_dim = reduce_dim

        if self.reduce_dim == False:
            self.ops = nn.Sequential(
                PointwiseConvolution2D(ch_in, ch_in*expansion, stride=1, padding=0, bias=True, dilation=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=int_bits, dec_bits=dec_bits),
                DepthwiseConvolution2D(ch_in*expansion, k_size=3, stride=1, padding=1, bias=True, dilation=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=int_bits, dec_bits=dec_bits),
                PointwiseConvolution2D(ch_in*expansion, ch_out, stride=1, padding=0, bias=True, dilation=1, with_bn=True, with_relu=False, quantization=quantization, int_bits=int_bits, dec_bits=dec_bits)
                )
        else:
            self.ops = nn.Sequential(
                PointwiseConvolution2D(ch_in, ch_in*expansion, stride=1, padding=0, bias=True, dilation=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=int_bits, dec_bits=dec_bits),
                DepthwiseConvolution2D(ch_in*expansion, k_size=3, stride=1, padding=0, bias=True, dilation=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=int_bits, dec_bits=dec_bits),
                PointwiseConvolution2D(ch_in*expansion, ch_out, stride=1, padding=0, bias=True, dilation=1, with_bn=True, with_relu=False, quantization=quantization, int_bits=int_bits, dec_bits=dec_bits)
                )

    def forward(self, x):
        # print(" INPUT: {}".format(x.shape))
        out = self.ops(x)
        # print(" OUTPUT: {}".format(out.shape))

        if self.reduce_dim == False:
            return out + x
        else:
            return out



class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features, bias=True, with_bn=False, with_relu=False, quantization=False, int_bits=0, dec_bits=0):
        super(FullyConnected, self).__init__()
        fc = nn.Linear(in_features, out_features, bias=bias)
        quantization = QuantizeActivation(int_bits=int_bits, dec_bits=dec_bits, quantization=quantization)
    
        if with_bn:
            if with_relu:
                self.operation = nn.Sequential(quantization, fc, quantization, nn.BatchNorm2d(out_features), nn.ReLU())
            else:
                self.operation = nn.Sequential(quantization, fc, quantization, nn.BatchNorm2d(out_features))
        else:
            if with_relu:
                self.operation = nn.Sequential(quantization, fc, quantization, nn.ReLU())
            else:
                self.operation = nn.Sequential(quantization, fc, quantization)

    def forward(self, inputs):
        outputs = self.operation(inputs)
        return outputs



class ResidualLayer(nn.Module):
    def __init__(self, ch_in, ch_out, skip_proj, quantization=False, int_bits=0, dec_bits=0):
        super(ResidualLayer, self).__init__()
        self.skip_proj = skip_proj

        if self.skip_proj == False:
            self.pathA = nn.Sequential(
                Convolution2D(ch_in, ch_out, k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=int_bits, dec_bits=dec_bits),
                Convolution2D(ch_out, ch_out, k_size=3, stride=1, padding=1, with_bn=True, with_relu=False, quantization=quantization, int_bits=int_bits, dec_bits=dec_bits)
            )

        else:
            self.pathA = nn.Sequential(
                Convolution2D(ch_in, ch_out, k_size=3, stride=2, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=int_bits, dec_bits=dec_bits),
                Convolution2D(ch_out, ch_out, k_size=3, stride=1, padding=1, with_bn=True, with_relu=False, quantization=quantization, int_bits=int_bits, dec_bits=dec_bits)
            )

            self.pathB = Convolution2D(ch_in, ch_out, k_size=1, stride=2, padding=0, with_bn=True, with_relu=False, quantization=quantization, int_bits=int_bits, dec_bits=dec_bits) 

        self.relu = nn.ReLU() 
        
    def forward(self, x):
        # print(" INPUT: {}".format(x.shape))
        out = self.pathA(x)
        # print(" OUTPUT: {}".format(out.shape))

        if self.skip_proj == False:
            out = out + x
        else:
            out = out + self.pathB(x)
        return self.relu(out)



class Inception(nn.Module):
    def __init__(self, ch_in, ch_out1, ch_out2A, ch_out2B, ch_out3A, ch_out3B, ch_out4, quantization=False, int_bits=0, dec_bits=0):
        super(Inception, self).__init__()
        self.branch1 = Convolution2D(ch_in, ch_out1, k_size=1, stride=1, padding=0, with_bn=True, with_relu=True, quantization=quantization, int_bits=int_bits, dec_bits=dec_bits)
                     
        self.branch2 = nn.Sequential(
            Convolution2D(ch_in, ch_out2A, k_size=1, stride=1, padding=0, with_bn=True, with_relu=True, quantization=quantization, int_bits=int_bits, dec_bits=dec_bits),
            Convolution2D(ch_out2A, ch_out2B, k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quantization=quantization, int_bits=int_bits, dec_bits=dec_bits)
        )
        self.branch3 = nn.Sequential(
            Convolution2D(ch_in, ch_out3A, k_size=1, stride=1, padding=0, with_bn=True, with_relu=True, quantization=quantization, int_bits=int_bits, dec_bits=dec_bits),
            Convolution2D(ch_out3A, ch_out3B, k_size=5, stride=1, padding=2, with_bn=True, with_relu=True, quantization=quantization, int_bits=int_bits, dec_bits=dec_bits)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Convolution2D(ch_in, ch_out4, k_size=1, stride=1, padding=0, with_bn=True, with_relu=True, quantization=quantization, int_bits=int_bits, dec_bits=dec_bits)
        )
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat((b1,b2,b3,b4), 1)



# class Xception_additive_block_entry(nn.Module):
#     def __init__(self, ch_IN, ch_OUT1, ch_OUT2, ch_OUT3, quant=True):
#         super(Xception_additive_block_entry, self).__init__()

#         self.entry_ops_pathA1 = nn.Sequential(
#             SeparableConvolution2D(ch_IN, ch_OUT1, k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quant=quant),
#             SeparableConvolution2D(ch_OUT1, ch_OUT1, k_size=3, stride=1, padding=1, with_bn=True, with_relu=False, quant=quant),
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#         )

#         self.entry_ops_pathB1 = nn.Sequential( Convolution2D(ch_IN, ch_OUT1, k_size=1, stride=1, padding=0, with_bn=True, with_relu=False, quant=quant) )

#         self.entry_ops_pathA2 = nn.Sequential(
#             SeparableConvolution2D(ch_OUT1, ch_OUT2, k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quant=quant),
#             SeparableConvolution2D(ch_OUT2, ch_OUT2, k_size=3, stride=1, padding=1, with_bn=True, with_relu=False, quant=quant),
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#         )

#         self.entry_ops_pathB2 = nn.Sequential( Convolution2D(ch_OUT1, ch_OUT2, k_size=1, stride=1, padding=0, with_bn=True, with_relu=False, quant=quant) )

#         self.entry_ops_pathA3 = nn.Sequential(
#             SeparableConvolution2D(ch_OUT2, ch_OUT2, k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quant=quant),
#             SeparableConvolution2D(ch_OUT2, ch_OUT3, k_size=3, stride=1, padding=1, with_bn=True, with_relu=False, quant=quant),
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#         )

#         self.entry_ops_pathB3 = nn.Sequential( Convolution2D(ch_OUT2, ch_OUT3, k_size=1, stride=1, padding=0, with_bn=True, with_relu=False, quant=quant) )

#         self.relu = nn.Sequential( nn.ReLU() )

        
#     def forward(self, x):
#         partialA1 = self.entry_ops_pathA1(x)
#         partialB1 = self.entry_ops_pathB1(x)
#         res1 =  self.relu(partialA1 + partialB1)

#         partialA2 = self.entry_ops_pathA2(res1)
#         partialB2 = self.entry_ops_pathB2(partialB1)
#         res2 =  self.relu(partialA2 + partialB2)

#         partialA3 = self.entry_ops_pathA3(res2)
#         partialB3 = self.entry_ops_pathB3(partialB2)
#         res3 =  self.relu(partialA3 + partialB3)
#         return res3


# class Xception_additive_block_middle(nn.Module):
#     def __init__(self, channels, quant=True):
#         super(Xception_additive_block_middle, self).__init__()
#         self.middle_ops = nn.Sequential(
#             SeparableConvolution2D(channels, channels, k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quant=quant),
#             SeparableConvolution2D(channels, channels, k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quant=quant),
#             SeparableConvolution2D(channels, channels, k_size=3, stride=1, padding=1, with_bn=True, with_relu=False, quant=quant)
#         )
#         self.relu = nn.Sequential( nn.ReLU() )
        
#     def forward(self, x):
#         partial = self.middle_ops(x)
#         res =  partial + x
#         return self.relu(res)




# class Xception_additive_block_exit(nn.Module):
#     def __init__(self, ch_IN, ch_OUT1, ch_OUT2, quant=True):
#         super(Xception_additive_block_exit, self).__init__()
#         self.exit_ops_pathA = nn.Sequential(
#             SeparableConvolution2D(ch_IN, ch_OUT1, k_size=3, stride=1, padding=1, with_bn=True, with_relu=True, quant=quant),
#             SeparableConvolution2D(ch_OUT1, ch_OUT2, k_size=3, stride=1, padding=1, with_bn=True, with_relu=False, quant=quant),
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#         )
#         self.exit_ops_pathB = nn.Sequential( Convolution2D(ch_IN, ch_OUT2, k_size=1, stride=1, padding=0, with_bn=True, with_relu=False, quant=quant) )
#         self.relu = nn.Sequential( nn.ReLU() )
        
#     def forward(self, x):
#         partialA = self.exit_ops_pathA(x)
#         partialB = self.exit_ops_pathB(x)
#         res =  partialA + partialB
#         return self.relu(res)