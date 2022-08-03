from utils import *
import glob

list = glob.glob('raccoon/raccoon/*.xml')
for i in list:
    convert_xml_annotation(i, 'yolo', 0)