import glob
import os
import xml.etree.ElementTree as ET

import numpy as np


def create_contour_from_ndpa(wsi, annotation_file):
    tree = ET.parse(annotation_file) 
    mpp_x=float(wsi.properties['openslide.mpp-x'])
    mpp_y=float(wsi.properties['openslide.mpp-y'])

    annotations=[] #list of contours

    for idx,ndpviewstate in enumerate(tree.getiterator('ndpviewstate')):
        annot_type = ndpviewstate.find('annotation').get('displayname')
        if annot_type == "AnnotateFreehandLine": continue #not closed

        if annot_type == 'AnnotateCircle':
            x = ndpviewstate.find('annotation').find('x').text
            y = ndpviewstate.find('annotation').find('y').text
            r = ndpviewstate.find('annotation').find('radius').text

            x = int(x)
            y = int(y)
            r = int(float(r)//(1000*mpp_x))
            x, y = conv(x, y, wsi, mpp_x, mpp_y)
            each_annot = (x,y,r)

        else:
            each_annot = []
            for point in ndpviewstate.find('annotation').find('pointlist'):
                x=int(point[0].text)
                y=int(point[1].text)
                x, y = conv(x, y, wsi, mpp_x, mpp_y)
                each_annot.append((x,y))
        annotations.append(np.array(each_annot, dtype=np.int))
            
    return annotations

def conv(x, y, wsi, mpp_x, mpp_y):
    openslide_x_nm_from_center=x-int(wsi.properties['hamamatsu.XOffsetFromSlideCentre'])
    openslide_y_nm_from_center=y-int(wsi.properties['hamamatsu.YOffsetFromSlideCentre'])
    openslide_x_nm_from_topleft=openslide_x_nm_from_center+int(wsi.properties['openslide.level[0].width'])*mpp_x*1000//2
    openslide_y_nm_from_topleft=openslide_y_nm_from_center+int(wsi.properties['openslide.level[0].height'])*mpp_y*1000//2
    openslide_x_pixels_from_topleft=openslide_x_nm_from_topleft//(1000*mpp_x)
    openslide_y_pixels_from_topleft=openslide_y_nm_from_topleft//(1000*mpp_y)
                
    return int(openslide_x_pixels_from_topleft), int(openslide_y_pixels_from_topleft)

def search_ndpa(wsifile, target="/wsi/analysis/CellType/ndpa"):
    all_ndpa = glob.glob(target+"/**/*.ndpa", recursive=True)
    for ndpa in all_ndpa:
        base_ndpa = os.path.basename(ndpa)
        if os.path.basename(wsifile) in base_ndpa:
            return ndpa

    return None
    
