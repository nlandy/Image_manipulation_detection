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
    isetpath = '/home/nlandy/Image_manipulation_detection/data/Columbia/Columbia/ImageSets/trainval.txt'
    seg_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    file = open(isetpath, 'w')

    for f in seg_files:
        name = f[0:-4]
        file.write(name + '\n')
    file.close()
