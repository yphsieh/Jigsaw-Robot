import matplotlib.pyplot as plt
import numpy as np
import cv2
import random as rd
from scipy import stats
import math
import os
import sys

def getKernel(ks):
    r = math.floor(ks/2)
    kernel = np.zeros((ks, ks), np.uint8)
    kernel[r][r] = 1
    for i in range(ks):
        for j in range(ks):
            if (i-r)**2 + (j-r)**2 <= r**2:
                kernel[i][j] = 1

def opening(img, kernel_size, itr):
    # kernel = np.ones((kernel_size,kernel_size), np.uint8)
    kernel = getKernel(kernel_size)
    img = cv2.erode(img, kernel, iterations=itr)
    img = cv2.dilate(img, kernel, iterations=itr)
    return img

def closing(img, kernel_size, itr):
    kernel = getKernel(kernel_size)
    img = cv2.dilate(img, kernel, iterations=itr)
    img = cv2.erode(img, kernel, iterations=itr)
    return img

def image_preprocess(img_path):
    img = cv2.imread(img_path)
    w, h = img[:, :, 0].shape
    img = img[int(0.2*w): int(0.95*w), int(0.085*h): int(0.9*h)]

    return img

def getRect(img):

    h, w = img.shape

    w_crop, crop_w, h_crop, crop_h = 10, 10, 10, 10
    while np.median(img[:,:h_crop]) == 0 and h_crop <= h*0.25: h_crop += 1
    while np.median(img[-crop_w:,:]) == 0 and crop_w <= w*0.25: crop_w += 1
    while np.median(img[:,-crop_h:]) == 0 and crop_h <= h*0.25: crop_h += 1
    while np.median(img[:w_crop,:]) == 0 and w_crop <= w*0.25: w_crop += 1

    img = img[w_crop:-crop_w,h_crop:-crop_h]
    return img

def remove_bg(im, thres=[10, 70, 50]):
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    med_h = stats.mode(im_hsv[:, :, 0], axis=None)[0][0]
    med_s = stats.mode(im_hsv[:, :, 1], axis=None)[0][0]
    med_v = stats.mode(im_hsv[:, :, 2], axis=None)[0][0]

    lower_thres = np.array([med_h-thres[0],med_s-thres[1],med_v-thres[2]])
    upper_thres = np.array([med_h+thres[0],med_s+thres[1],med_v+thres[2]])

    masked_img = cv2.inRange(im_hsv, lower_thres, upper_thres)
    cv2.imwrite('images/tmp/0.jpg',masked_img)

    masked_img = closing(masked_img, 5, 1)
    cv2.imwrite('images/tmp/1.jpg',masked_img)
    masked_img = closing(masked_img, 3, 2)
    cv2.imwrite('images/tmp/2.jpg',masked_img)
    # masked_img = opening(masked_img, 3, 2)
    # cv2.imwrite('images/tmp/3.jpg',masked_img)
    
    res_img = cv2.bitwise_and(im, im, mask = (255-masked_img))
    tmp = np.repeat(masked_img[:,:,np.newaxis], 3, axis = 2)
    remove_bg = cv2.bitwise_or(res_img, tmp)

    cv2.imwrite('images/test/backgroundRemoved.jpg', remove_bg)
    return masked_img, res_img #, remove_bg

def detect_pieces(im, name, thres=[10, 70, 50]):

    if not os.path.isdir("./results/" + name):
        print("creating folder './results/" + name + "'")
        os.mkdir("./results/" + name)
        os.mkdir("./results/" + name + "/cropped")
    
    masked_img, res_img = remove_bg(im, thres)
    cv2.imwrite("./results/" + name + '/res.jpg', res_img)
    contours, _ = cv2.findContours(masked_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    display = im
    total_area = masked_img.shape[0] * masked_img.shape[1]
    candidate_box = []
    crop_pieces = []

    for i, cnt in enumerate(contours):
        # print(i)
        area = cv2.contourArea(cnt)
        if area > total_area*0.002 and area < total_area*0.2:
            cv2.drawContours(display, [cnt], -1, (0,0,255), 5)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            candidate_box.append(box)
            cv2.drawContours(display, [box], -1, (0, 255, 0), 3)
            cropped = removeShadow(crop(res_img, rect, box))
            crop_pieces.append(cropped)
            cv2.imwrite("./results/" + name + f"/cropped/crop_{i:02d}.jpg", cropped)
    cv2.imwrite("./results/" + name + "/display.jpg", display)
    return crop_pieces

def detect_corners(img, numCorners=4):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, numCorners,0.1,40)
    corners = np.int0(corners)
    print(len(corners))
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,(0, 255, 255),2)
    cv2.imwrite('images/tmp/corners' + str(len(corners)) +'.jpg',img)
    return corners

def removeShadow(img):
    rgb_planes = cv2.split(img)

    result_norm_planes = []
    for plane in rgb_planes:
        norm_img = cv2.normalize(plane, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        norm_img = cv2.equalizeHist(norm_img)
        result_norm_planes.append(norm_img)

    result_norm = cv2.merge(result_norm_planes)
    return result_norm

def crop(img, rect, box):
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped = cv2.warpPerspective(img, M, (width, height))
    return cropped

if __name__=='__main__':
    img = image_preprocess(sys.argv[1])
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    