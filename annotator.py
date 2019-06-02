from os import listdir
from os.path import isfile, join
mypath = '/home/nlandy/Image_manipulation_detection/data/Columbia/Columbia/SegmentationObject/'
annopath = '/home/nlandy/Image_manipulation_detection/data/Columbia/Columbia/Annotations/'
seg_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

from random import randint
from PIL import Image
import numpy as np
from lib.datasets.factory import get_imdb
from lib.datasets.xml_op import *
import xml.etree.ElementTree as ET
from shutil import copyfile


def create_xml(savefile, xmin, ymin, xmax, ymax):
    def create_node(tag, property_map, content):
        element = Element(tag, property_map)
        element.text = content
        return element
    open(savefile, 'r')
    close(savefile)

    tree = ET.parse(savefile)
    root = tree.getroot()
    for obj in root.findall('object'):
        root.remove(obj)
    new_obj = Element('object', {})
    new_obj.append(create_node('name', {}, 'tampered'))
    bndbox = Element('bndbox', {})
    bndbox.append(create_node('xmin', {}, str(xmin)))
    bndbox.append(create_node('ymin', {}, str(ymin)))
    bndbox.append(create_node('xmax', {}, str(xmax)))
    bndbox.append(create_node('ymax', {}, str(ymax)))
    new_obj.append(bndbox)
    root.append(new_obj)
    tree.write(savefile)

for f in seg_files:
    segpath = mypath + f
    name = f[0:-4]
    annopath_f = annopath + name + '.xml'

    print(segpath)

    seg = Image.open(segpath)
    seg.convert('RGB')
    seg_np = np.asarray(np)
    print(seg_np.shape)
    seg_np_red = seg_np[:,:,0]
    indices = np.where(seg_np_red == 0)
    minx = np.min(indices[1])
    maxx = np.max(indices[1])
    miny = np.min(indices[0])
    maxy = np.max(indices[0])

    create_xml(annopath_f, xmin, ymin, xmax, ymax)
