import torch
from torch.autograd import Variable
import numpy as np
import shutil
import os
import copy
from time import perf_counter
from torch import nn
# from torch.nn import Module, Sequential
import model_zoo.models as models


__all__ = ['train', 'test','PerLayerQuantization', 'count_layers', 'set_acts_quant', 'get_MAC_per_layer', 'set_model_act_bitwidth', 'print_model_act_bitwidth']


def train(train_loader, model, criterion, optimizer, weights_bitwidth, device, quantization=False):
    model.train()
    start = perf_counter()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if quantization:
            PerLayerQuantization(model, weights_bitwidth)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        res = []
        for k in topk:
            _, pred = output.topk(k, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def test(test_loader, model, weights_bitwidth, device, criterion=None, quantization=False):
    top1, top5, test_loss = 0, 0, 0
    if isinstance(model, list):
        for inst in model:
            inst.eval()
    else:
        model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = None
            if isinstance(model, list):
                for inst in model:
                    if quantization:
                        PerLayerQuantization(inst, weights_bitwidth)
                    if output is None:
                        output = inst(data)
                    else:
                        output += inst(data)
            else:
                if quantization:
                    PerLayerQuantization(model, weights_bitwidth)
                output = model(data)
            
            if(criterion is not None):
                test_loss += criterion(output, target).item()
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            top1 += prec1[0]
            top5 += prec5[0]
    top1_acc = 100. * top1 / len(test_loader.sampler)
    top5_acc = 100. * top5 / len(test_loader.sampler)
    if (criterion is not None):
        test_loss /= len(test_loader.sampler)
        return top1_acc, test_loss
    return top1_acc, top5_acc


###############################################################################################
######################### QUANTIZATION ########################################################
###############################################################################################
def quantize(module, step, max_val, min_val):
    with torch.no_grad():
        module.weight.data = torch.round(module.weight/step)*step
        module.weight[module.weight > max_val].data = torch.tensor(max_val, dtype=float)
        module.weight[module.weight < min_val].data = torch.tensor(min_val, dtype=float)
        module.bias.data = torch.round(module.bias/step)*step
        module.bias[module.bias > max_val].data = torch.tensor(max_val, dtype=float)
        module.bias[module.bias < min_val].data = torch.tensor(min_val, dtype=float)

def compute_quantization_params(n_layers, nbits):
    max_val = np.zeros(n_layers, dtype=np.float32)
    min_val = np.zeros(n_layers, dtype=np.float32)
    step = np.ones(n_layers, dtype=np.float32)
    for l in range(n_layers):
        step[l] = 1 / (2 ** (nbits[l]-1))
        max_val[l] = 1 - step[l] 
        min_val[l] = -max_val[l]
    return max_val, min_val, step

def PerLayerQuantization(model, nbits):
    if nbits[0] == 0:
        return
    n_layers = int(len(nbits))
    layer = 0
    max_val, min_val, step = compute_quantization_params(n_layers, nbits)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            with torch.no_grad():
                quantize(module, step[layer], max_val[layer], min_val[layer])
            layer = layer + 1
###############################################################################################
###############################################################################################
###############################################################################################

def set_acts_quant(model, act_int_bits):
    for name, module in model.named_modules():
        if (module.__class__.__name__ == "QuantizeActivation"):
            module.set_int_bits(act_int_bits)
            module.set_quantization()


def count_layers(model):
    conv_layers = 0
    fc_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers = conv_layers + 1
        elif isinstance(module, nn.Linear):
            fc_layers = fc_layers + 1
    return conv_layers+fc_layers, conv_layers, fc_layers




def get_MAC_per_layer(model, output_features_size):
    MACs = []
    i = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            MACs.append(np.prod(module.weight.shape)*output_features_size[i]*output_features_size[i])
            i = i + 1
        if isinstance(module, nn.Linear):
            MACs.append(np.prod(module.weight.shape))
    return MACs

def set_model_act_bitwidth(model, layer, bitwidth):
    l = 0
    for name, module in model.named_modules():
        if (module.__class__.__name__ == "QuantizeActivation"):
            if(layer == l):
                if(bitwidth==16):
                    module.set_int_bits(7)
                    module.set_dec_bits(8)
                if(bitwidth==8):
                    module.set_int_bits(4)
                    module.set_dec_bits(3)

                return model
            l = l + 1

def print_model_act_bitwidth(model):
    l = 0
    for name, module in model.named_modules():
        if (module.__class__.__name__ == "QuantizeActivation"):
            l = l + 1
            print("Layer {}: {} bits".format(l, module.get_act_bits()))
