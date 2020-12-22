import matplotlib.pyplot as plt
import numpy as np
import cv2
import random as rd
from numpy.lib.function_base import disp
from scipy import stats
import math
import os
import sys
from scipy import stats

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
    mid_points = []
    i = 0
    for cnt in contours:
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
            mid = detect_middle(cnt.squeeze(), rect, display, name, i)
            mid_points.append(mid)
            cv2.circle(display, (int(mid[0]),int(mid[1])), radius=5, color=(255, 0, 0), thickness=10)
            cv2.imwrite("./results/" + name + f"/cropped/crop_{i:02d}.jpg", cropped)
            i+=1
    cv2.imwrite("./results/" + name + "/display.jpg", display)
    return crop_pieces, mid_points

def detect_middle(cnt, box, img, name, i, vis=False):
    assert(cnt.shape[1]==2)
    mid = box[0]
    size = box[1]
    theta = (box[2])*math.pi/180
    
    num = cnt.shape[0]
    cnt = cnt-mid
    
    r = np.array(( (np.cos(theta), -np.sin(theta)),
                   (np.sin(theta),  np.cos(theta)) ))
    
    cnt = np.matmul(cnt, r).astype(np.int16)
    # np.savetxt(f'cnt{int(mid[0])}.txt', cnt, fmt='%d')
    
    up = []
    down = []
    left = []
    right= []
    thres=[size[0]/4, size[1]/4]

    count = 0
    while not (up and down):
        if count > 5:
            return [0, 0]
        for p in cnt:
            if p[0]<-size[1]/2+thres[1]:
                up.append(p)
            elif p[0]>size[1]/2-thres[1]:
                down.append(p)
            if p[1]<-size[0]/2+thres[0]:
                left.append(p)
            elif p[1]>size[0]/2-thres[0]:
                right.append(p)
        thres += [20, 20]
        count += 1
    
    if vis:
        for p in up:
            p = r.dot(p)
            px = int(p[0] + mid[0])
            py = int(p[1] + mid[1])
            cv2.circle(img, (px, py), radius=5, color=(255, 0, 0), thickness=10)
        for p in down:
            p = r.dot(p)
            px = int(p[0] + mid[0])
            py = int(p[1] + mid[1])
            cv2.circle(img, (px, py), radius=5, color=(125, 125, 0), thickness=10)
        for p in left:
            p = r.dot(p)
            px = int(p[0] + mid[0])
            py = int(p[1] + mid[1])
            cv2.circle(img, (px, py), radius=5, color=(125, 0, 125), thickness=10)
        for p in right:
            p = r.dot(p)
            px = int(p[0] + mid[0])
            py = int(p[1] + mid[1])
            cv2.circle(img, (px, py), radius=5, color=(0, 125, 125), thickness=10)
        cv2.imwrite("./results/" + name + f"/test_mid.jpg", img)
    
    up_left = sorted(up, key=lambda x: x[0]+x[1])[0]
    up_right = sorted(up, key=lambda x: x[0]-x[1])[0]
    down_left = sorted(down, key=lambda x: -x[0]+x[1])[0]
    down_right = sorted(down, key=lambda x: -x[0]-x[1])[0]
    
    mid_point = (up_left+up_right+down_left+down_right)//4
    mid_point = r.dot(mid_point)
    return mid_point+mid

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
    