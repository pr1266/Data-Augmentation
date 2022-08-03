from utils import *
import glob

list = glob.glob('raccoon/*.xml')
for i in list:
    convert_xml_annotation(i, 'yolo', 0)