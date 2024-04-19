import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.utils_baseline import get_network


def print_parameters():
    net = get_network("ConvNet", 3, 10)
    for mn, m in net.named_modules():
        print("Module Name: {}".format(mn))
        for n, p in m.named_parameters(recurse=False):
            print("Parameter Name: {}".format(n))
            print("Parameter Count: {}".format(p.numel()))
            print("Parameter Size: {}".format(p.size()))


if __name__ == '__main__':
    print_parameters()