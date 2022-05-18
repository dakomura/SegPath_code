import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import stain


def adjust_pos(scale, xx, yy):
    s = 2 ** scale
    modx = xx % s
    mody = yy % s
    return xx - modx, yy - mody

def threshold_otsu(gray, min_value=0, max_value=255):
    hist = [np.sum(gray == i) for i in range(256)]
    s_max = (0,-10)
    for th in range(256):
        
        n1 = sum(hist[:th])
        n2 = sum(hist[th:])
        
        if n1 == 0 : mu1 = 0
        else : mu1 = sum([i * hist[i] for i in range(0,th)]) / n1   
        if n2 == 0 : mu2 = 0
        else : mu2 = sum([i * hist[i] for i in range(th, 256)]) / n2
        s = n1 * n2 * (mu1 - mu2) ** 2
        if s > s_max[1]:
            s_max = (th, s)
    
    t = s_max[0]
    return t

def get_contour2(op, sc=4, kar=81, large=False):
    scale = sc
    w, h = op.dimensions

    wmin, hmin = w//5, h//5
    wmax, hmax = (4*w)//5, (4*h)//5
    wmin, hmin = wmin // (2 ** scale), hmin // (2 ** scale)
    wmax, hmax = wmax // (2 ** scale), hmax // (2 ** scale)

    rhe = op.read_region([0, 0], scale, [int(w / (2 ** scale)), int(h / (2 ** scale))])
    img = np.array(rhe)[:, :, [2, 1, 0]] 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Otsu's thresholding after Gaussian filtering
    blur_img = cv2.GaussianBlur(img_gray, (kar, kar), 0)
    t = threshold_otsu(blur_img[wmin:wmax, hmin:hmax])
    
    par = [1.1, 1.05, 1.02, 1.0]
    rec = 1.1
    ma_area = 0
    
    for p in par:
        tt = int(t * p)
        _, th = cv2.threshold(blur_img, tt, 255, cv2.THRESH_BINARY)
    
        contours, _ = cv2.findContours(th,
                                   cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
        spot_min = 12800000 / (4 ** scale)

        if large:
            spot_max = float('inf')
        else:
            spot_max = 256000000 / (4 ** scale)
    
        area = 0
    
        for c in contours:
            if(spot_min < cv2.contourArea(c) < spot_max):
                area += (cv2.contourArea(c)) * (4 ** scale)
        if(ma_area < area):
            ma_area = area
            rec = p
     
    t = int(t*rec)
    _, th = cv2.threshold(blur_img, t, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(th,
                                   cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
    spot_min = 12800000 / (4 ** scale)
    if large:
        spot_max = float('inf')
    else:
        spot_max = 256000000 / (4 ** scale)
    
    contours_orig = [c for c in contours if spot_min < cv2.contourArea(c) < spot_max]
    contours = [c * (2 ** scale) for c in contours_orig]
    img_contour = cv2.drawContours(cv2.UMat(img), contours_orig, -1, (0, 255, 0), 6)
    img_contour = cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB)
    if type(img_contour) != np.ndarray:
        img_contour = img_contour.get()

    return contours, img_contour, t


def get_contour(op, scale=4):
    w, h = op.dimensions
    rhe = op.read_region([0, 0], scale, [int(w / (2 ** scale)), int(h / (2 ** scale))])
    img = np.array(rhe)[:, :, [2, 1, 0]]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding after Gaussian filtering
    blur_img = cv2.GaussianBlur(img_gray, (9, 9), 0)
    _, th = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(th,
                                   cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
    contours_orig = [c for c in contours if 50000 < cv2.contourArea(c) < 1000000]
    contours = [c * (2 ** scale) for c in contours_orig]

    img_contour = cv2.drawContours(img, contours_orig, -1, (0, 255, 0), 6)
    img_contour = cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB)
    if type(img_contour) != np.ndarray:
        img_contour = img_contour.get()

    return contours, img_contour





def get_region(op_ihc, op_he, pos, size, scale=1, is_show=False, shift=(0, 0)):
    (px, py) = adjust_pos(scale, pos[0] + shift[0], pos[1] + shift[1])
    (hpx, hpy) = adjust_pos(scale, pos[0], pos[1])

    region_ihc = op_ihc.read_region([px, py], scale, size)
    region_he = op_he.read_region([hpx, hpy], scale, size)

    region_ihc = 255 - np.array(region_ihc)[:, :, 0]

    if size[0] > 500:
        region_he = stain.extract_H(region_he, split=True)
    else:
        region_he = stain.extract_H(region_he, split=False)

    if is_show:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(region_ihc, cmap="binary")
        ax1.axis('off')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(region_he, cmap="binary")
        ax2.axis('off')

    return region_ihc, region_he



def get_region_he(op_ihc, op_he, pos, size, scale=1, shift=(0, 0)):
    (px, py) = adjust_pos(scale, pos[0] + shift[0], pos[1] + shift[1])
    (hpx, hpy) = adjust_pos(scale, pos[0], pos[1])

    region_ihc = op_ihc.read_region([px, py], scale, size)
    region_he = op_he.read_region([hpx, hpy], scale, size)

    region_ihc = 255 - np.array(region_ihc)[:, :, 0]

    region_he = np.array(region_he)

    return region_ihc, region_he

def is_overlap(pos, imgsize, annot):
    if annot == None:
        return False, None
    x1, y1 = pos
    x2, y2 = x1 + imgsize, y1 + imgsize
    for an in annot:
        if an.ndim == 1: #circle
            overlap =  checkOverlap_circle(an[2], an[0], an[1],
                                           x1, y1, x2, y2)
            if overlap: return True, an

        if an.ndim == 2: #polygon
            overlap =  checkOverlap_contour(an, x1, y1, x2, y2)
            if overlap: return True, an

    return False, None

def checkOverlap_contour(contour, X1, Y1, X2, Y2):
    for x in range(X1, X2, 100):
        for y in range(Y1, Y2, 100):
            pos = (x,y)
            if cv2.pointPolygonTest(contour, pos, False) > 0:
                return True

    return False

def checkOverlap_circle(R, Xc, Yc, X1, Y1, X2, Y2):

    Xn = max(X1, min(Xc, X2))
    Yn = max(Y1, min(Yc, Y2))

    Dx = Xn - Xc
    Dy = Yn - Yc
    return (Dx**2 + Dy**2) <= R**2
