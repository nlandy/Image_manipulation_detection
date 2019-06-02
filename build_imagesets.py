from os import listdir
from os.path import isfile, join
import os

from random import randint
from PIL import Image
import numpy as np
from lib.datasets.factory import get_imdb
from lib.datasets.xml_op import *
import xml.etree.ElementTree as ET
from shutil import copyfile

if __name__ == '__main__':
    mypath = '/home/nlandy/Image_manipulation_detection/data/Columbia/Columbia/SegmentationObject/'
    annopath = '/home/nlandy/Image_manipulation_detection/data/Columbia/Columbia/Annotations/'
    isetpath = '/home/nlandy/Image_manipulation_detection/data/Columbia/Columbia/ImageSets/Main/trainval.txt'
    trainpath = '/home/nlandy/Image_manipulation_detection/data/Columbia/Columbia/ImageSets/Main/train.txt'
    valpath = '/home/nlandy/Image_manipulation_detection/data/Columbia/Columbia/ImageSets/Main/val.txt'
    seg_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    NUM_TRAIN = 150
    ctr = 0

    file = open(isetpath, 'w')
    file_train = open(trainpath, 'w')
    file_val = open(testpath, 'w')

    for f in seg_files:
        name = f[0:-15]
        file.write(name + '\n')
        if ctr < NUM_TRAIN:
            file_train.write(name + '\n')
        else:
            file_val.write(name + '\n')
        print('Image ' + str(cnt) + '...\n')
        ctr += 1
    file.close()
    file_train.close()
    file_val.close()
