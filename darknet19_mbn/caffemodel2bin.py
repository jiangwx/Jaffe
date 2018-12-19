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
parser.add_argument('--path', type=str, default='.', help='path to save the binary file')
args = parser.parse_args()

if not os.path.isfile(args.prototxt):
    print 'please specify prototxt'
    exit()
if not os.path.isfile(args.caffemodel):
    print 'please specify caffemodel'
    exit()
if not os.path.exists(args.path+'/weight'):
    os.makedirs(args.path+'/weight')
if not os.path.exists(args.path+'/blob'):
    os.makedirs(args.path+'/blob')

prototxt=args.prototxt
caffemodel=args.caffemodel
path=args.path

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(prototxt, caffemodel, caffe.TEST)
net.forward()
layers = list(net.blobs)

for layer in layers:
    if(layer.find('conv')!=-1):
        if(layer.find('relu')!=-1):
            print layer + ' skip'
        else:
            if len(net.params[layer])==1:
                wt=open(path+'/weight/'+ layer +'.wt','w')
                weight = net.params[layer][0].data
                weight.tofile(wt)
                print layer + ' weight saved to ' + path+'/weight/'+ layer +'.wt'
            elif len(net.params[layer])==2:
                wt=open(path+'/weight/'+ layer +'.wt','w')
                weight = net.params[layer][0].data
                weight.tofile(wt)
                bs=open(path+'/weight/'+layer+'.bs','w')
                bias = net.params[layer][1].data
                bias.tofile(bs)
                print layer + ' weight saved to ' + path+'/weight/'+ layer +'.wt, bias saved to ' + path+'/weight/'+ layer +'.bs '

for layer in layers:
    if (layer.find('label')!=-1)|(layer.find('accuracy')!=-1)|(layer.find('loss')!=-1)|(layer.find('split')!=-1):
        print 'skip ' + layer
    elif (layer.find('relu')!=-1):
        bb = open(path+'/blob/'+layer[:layer.rfind('_')]+'.bb','w')
        blob = net.blobs[layer].data[0]
        blob.tofile(bb)
        print layer +' data saved to ' + path +'/blob/'+ layer[:layer.rfind('_')] +'.bb'
    elif (layer.find('pool')!=-1)|(layer.find('data')!=-1):
        bb = open(path+'/blob/'+layer+'.bb','w')
        blob = net.blobs[layer].data[0]
        blob.tofile(bb)
        print layer +' data saved to ' + path +'/blob/'+ layer+'.bb'
