from cProfile import label
from calendar import leapdays
from fileinput import filename
import os
import argparse
import shutil
from lxml import etree
from tqdm import tqdm
from io import StringIO

import cv2
from bs4 import BeautifulSoup
import numpy as np
import ast
from math import *
import time
#from insight_utils import *
from skimage import transform as trans
import os
import json
import re
from numpy import array
#app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection'])
#app.prepare(ctx_id=0, det_size=(640, 640))

#rec_model_path = "./insightface/models/buffalo_l/w600k_r50.onnx"
#handler = insightface.model_zoo.get_model(rec_model_path)
#handler.prepare(ctx_id=0)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
        
def txt_to_json(ground_truth, filename):
    txt_filename = filename[:-3] + "txt"
    with open(os.path.join(ground_truth, txt_filename)) as f:
        lol = f.read()
        lol2 = str(lol).replace("'", '"')

        sep1 = ', "Bbox"'
        lol3 = lol2.split(sep1, 1)[0]
        lol1 = lol2.split(sep1, 1)[1]
        
        lol1 = str(lol1).replace(" ", "")
        lol1 = str(lol1).replace("\n", "")
        lol1 = str(lol1).replace("],]", "]]")
        lol1 = str(lol1).replace("array(", '')
        lol1 = str(lol1).replace(",dtype=float32)", '')
        #lol1 = str(lol1).replace("dtype=float32),", '')
        lol1 = str(lol1).replace(".,", '.0,')
        lol1 = str(lol1).replace(".]", '.0]')
        sep = ',"Landmarks"'
        lol1 = lol1.split(sep, 1)[0]
        lol1 = lol1 + str("}")
        lol1 = str(lol1).replace("[[]]", '[[0]]')
        #print("lol3: ", lol3)
        #print("lol1: ", lol1)
        lol4 = lol3 + sep1 + lol1
        #print("lol4: ", lol4)
        #lol1 = str(lol1).replace("dtype=float32)", '')
        original_path = os.getcwd()
        os.chdir(ground_truth)
        #print("Inferred Result: ", lol1)
        json_filename = filename[:-3] + "json"
        with open(json_filename, "w") as outfile:
                outfile.write(lol4)
        os.chdir(original_path)

#***********************************************************
#                Testing
#***********************************************************
##***********************************************************
#                  Match manual annotation 
#************************************************************

def parse_anno_file(cvat_xml, image_name):
    root = etree.parse(cvat_xml).getroot()
    anno = []
    image_name_attr = ".//image[@name='{}']".format(image_name)
    for image_tag in root.iterfind(image_name_attr):
        image = {}
        for key, value in image_tag.items():
            image[key] = value
        image['shapes'] = []
        for poly_tag in image_tag.iter('polygon'):
            polygon = {'type': 'polygon'}
            for key, value in poly_tag.items():
                polygon[key] = value
            image['shapes'].append(polygon)
        for box_tag in image_tag.iter('box'):
            box = {'type': 'box'}
            for key, value in box_tag.items():
                box[key] = value
            box['points'] = "{0},{1};{2},{1};{2},{3};{0},{3}".format(
                box['xtl'], box['ytl'], box['xbr'], box['ybr'])
            image['shapes'].append(box)
        image['shapes'].sort(key=lambda x: int(x.get('z_order', 0)))
        anno.append(image)
    #print("Parsed XML File", anno)
    return anno

def genrate_final_annotation(ground_truth , xml_file, images):
    tot_count = 0
    counting_fails = 0
    for filename in os.listdir(ground_truth):
        tot_count += 1
        final=[]
        labels_final =[]
        txt_to_json(ground_truth, filename)
        with open(os.path.join(ground_truth ,filename[:-3]+"json")) as f:
            #print("Channel: ", ground_truth)
            #print("Filename: ", filename)
            data1 = json.load(f)
            #print(data1)
            bbox = data1["Bbox"]
            labelss = data1["Label"]
            found = 0
            for filename1 in os.listdir(images):
                if (filename[:-3]==filename1[:-3]):
                    counting_fails += 1
                    found = 1
                    #print("Filename 1: ", filename1)
                    anno = parse_anno_file(xml_file , filename1)
                    annotation = anno[0]['shapes']
                    #print("Annotations: ", annotation)
                    #ov_boxes = []
                    #ov_labels = []
                    for j in range(len(annotation)):
                        box_value = annotation[j]['points']
                        labels = annotation[j]['label']
                        #print("Manual Labels: ", labels)
                        #print("Manual Box Values: ", box_value)
                        box_value = box_value.split(";")
                        x1 =box_value[0].replace("\'" ,"").split(',')
                        y1= box_value[1].replace("\'" ,"").split(',')
                        x2= box_value[2].replace("\'" ,"").split(',')
                        y2= box_value[3].replace("\'" ,"").split(',')
                        x11= float(x1[0])
                        y11=float(x1[1])
                        x12 = float(y1[0])
                        y12= float(y2[1])
                        boxx1=[x11,y11,x12,y12]
                        #print("boxx1: ", boxx1)
                        #print("bbox: ", bbox)
                        #print("Length of bbox: ", len(bbox))
                        labels_final.append(labels)
                        final.append(boxx1)
            if found == 0:
                #print("Image is Gtg")
                final= bbox
                labels_final= labelss

            results = {"Label":labels_final, "Bbox":final}
            json_object = json.dumps(results , cls=NumpyEncoder ,  indent = 2)
            filename =os.path.join(os.getcwd(),ground_truth[:-15],"infered_results_final" , filename[:-3] + "json")
            with open(filename, "w") as outfile:
                outfile.write(json_object)
    print("Total Frames: ", tot_count)
    print("Failed Inferences: ", counting_fails)
            #for i in range(data1["Bbox"]):
    return

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou