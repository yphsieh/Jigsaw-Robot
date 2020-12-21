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

def image_preprocess(img):
    w, h = img[:, :, 0].shape
    img = img[int(0.15*w): int(0.9*w), int(0.15*h): int(0.85*h)]
    return img

def remove_bg(im, thres):
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    med_h = stats.mode(im_hsv[:, :, 0], axis=None)[0][0]
    med_s = stats.mode(im_hsv[:, :, 1], axis=None)[0][0]
    med_v = stats.mode(im_hsv[:, :, 2], axis=None)[0][0]

    lower_thres = np.array([med_h-thres[0],med_s-thres[1],med_v-thres[2]])
    upper_thres = np.array([med_h+thres[0],med_s+thres[1],med_v+thres[2]])

    masked_img = cv2.inRange(im_hsv, lower_thres, upper_thres)

    masked_img = closing(masked_img, 5, 3)
    masked_img = closing(masked_img, 25, 3)
    masked_img = opening(masked_img, 5, 2)
    masked_img = cv2.dilate(masked_img, getKernel(15), iterations=2)
    
    res_img = cv2.bitwise_and(im, im, mask = (255-masked_img))
    # tmp = np.repeat(masked_img[:,:,np.newaxis], 3, axis = 2)
    # remove_bg = cv2.bitwise_or(res_img, tmp)

    cv2.imwrite('images/test/backgroundRemoved.jpg', res_img)
    return masked_img, res_img

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
            cropped = crop(res_img, rect, box)
            crop_pieces.append(cropped)
            cv2.imwrite("./results/" + name + f"/cropped/crop_{i:02d}.jpg", cropped)
    cv2.imwrite("./results/" + name + "/display.jpg", display)
    return crop_pieces

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
    im = cv2.imread(sys.argv[1])
    im = image_preprocess(im)
    cv2.imwrite('./results/test/crop.jpg', im)
    detect_pieces(im, )
    