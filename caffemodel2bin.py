
import caffe
import numpy as np
from numpy import array
from numpy.random import normal
from matplotlib import pyplot
import os, sys
import struct
import argparse

parser = argparse.ArgumentParser(description='caffemodel to binary file')
parser.add_argument('--prototxt', type=str, default='', help='caffe prototxt')
parser.add_argument('--caffemodel', type=str, default='', help='caffe caffemodel')
parser.add_argument('--path', type=str, default='weight', help='path to save the binary file')
args = parser.parse_args()

if not os.path.isfile(args.prototxt):
    print 'please specify prototxt'
    exit()
if not os.path.isfile(args.caffemodel):
    print 'please specify caffemodel'
    exit()
if not os.path.exists(args.path):
    os.makedirs(args.path)

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

blob = list(net.blobs)
for layer in blob:
    if(layer.find('conv')!=-1):
        if(layer.find('bn')!=-1):
            mn=open(args.path + '/'+layer[:layer.rfind('_')]+'.mn','w')
            mean=net.params[layer][0].data
            mean.tofile(mn)
            print layer + ' mean saved to ' + args.path + '/'+ layer[:layer.rfind('_')] +'.mn'
            variance=net.params[layer][1].data
        elif(layer.find('scale')!=-1):
            bs=open(args.path + '/'+layer[:layer.rfind('_')]+'.bs','w')
            bias = net.params[layer][1].data
            bias.tofile(bs)
            print layer + ' scale and bias saved to ' + args.path + '/'+ layer[:layer.rfind('_')] +'.bs'
            scale = net.params[layer][0].data
            variance=1/np.sqrt(variance)
            variance_scale=np.multiply(scale, variance)
            vs=open(args.path + '/'+layer[:layer.rfind('_')]+".vs",'w')
            variance_scale.tofile(vs)
            print layer[:layer.rfind('_')] + ' scale x variance saved to ' + args.path + '/' + layer[:layer.rfind('_')] +'.vs'
        elif(layer.find('relu')!=-1):
            print layer + ' skip'
        else:
            wt=open(args.path + '/'+ layer +'.wt','w')
            weight = net.params[layer][0].data
            weight.tofile(wt)
            print layer + ' weight saved to ' + args.path + '/'+ layer +'.wt'
