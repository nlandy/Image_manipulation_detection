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


def create_xml(savefile, xmin, ymin, xmax, ymax):
    def create_node(tag, property_map, content):
        element = Element(tag, property_map)
        element.text = content
        return element
    file = open(savefile, 'w')

    anno = Element('annotation', {})
    new_obj = ET.SubElement(anno, 'object')
    new_obj.append(create_node('name', {}, 'tampered'))
    bndbox = ET.SubElement(anno, 'bndbox')
    bndbox.append(create_node('xmin', {}, str(xmin)))
    bndbox.append(create_node('ymin', {}, str(ymin)))
    bndbox.append(create_node('xmax', {}, str(xmax)))
    bndbox.append(create_node('ymax', {}, str(ymax)))
    new_obj.append(bndbox)

    anno_str = ET.tostring(anno).decode("utf-8")
    new_obj_str = ET.tostring(new_obj).decode("utf-8")
    bnd_box_str = ET.tostring(bndbox).decode("utf-8")

    print(new_obj_str)
    print(bnd_box_str)

    file.write(anno_str)

    file.close()


if __name__ == '__main__':
    mypath = '/home/nlandy/Image_manipulation_detection/data/Columbia/Columbia/SegmentationObject/'
    annopath = '/home/nlandy/Image_manipulation_detection/data/Columbia/Columbia/Annotations/'
    seg_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for f in seg_files:
        segpath = mypath + f
        name = f[0:-15]
        annopath_f = annopath + name + '.xml'

        print(segpath)

        #print(os.path.isfile(segpath))

        seg = Image.open(segpath)
        #print(seg)
        seg.convert('RGB')
        #print(seg)
        seg_np = np.asarray(seg)
        #print(seg_np.shape)
        seg_np_red = seg_np[:,:,0]
        indices = np.where(seg_np_red == 0)
        xmin = np.min(indices[1])+1
        xmax = np.max(indices[1])+1
        ymin = np.min(indices[0])+1
        ymax = np.max(indices[0])+1

        create_xml(annopath_f, xmin, ymin, xmax, ymax)
