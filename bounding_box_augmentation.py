from cv2 import cv2
from matplotlib import pyplot as plt
import albumentations as A
import glob
import os

os.system('cls')

img_path = 'teeth_data/279.jpg'
txt_path = 'teeth_data/279.txt'

lines = []
with open(txt_path, 'r') as F:
    lines = F.readlines()

for i in lines:
    



