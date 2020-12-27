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
    img = img[int(0.2*w): int(0.95*w), int(0.085*h): int(0.9*h)]
    return img

def remove_bg(im, thres=[5, 80, 70]):
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

def detect_pieces(im, name, thres=[5, 95, 70]):
    masked_img, res_img = remove_bg(im, thres)
    cv2.imwrite("./results/" + name + '/res.jpg', res_img)
    contours, _ = cv2.findContours(masked_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    display = im
    total_area = masked_img.shape[0] * masked_img.shape[1]
    candidate_box = []
    crop_pieces = []
    mid_points = []
    corners = []
    crops = []
    angles = []
    i = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > total_area*0.002 and area < total_area*0.2:
            cv2.drawContours(display, [cnt], -1, (0,0,255), 5)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            candidate_box.append(box)
            cropped = removeShadow(crop(im, rect, box))
            crop_pieces.append(cropped)
            mid, corner, inner, angle, l, r = detect_middle(cnt.squeeze(), rect, im, name, i)
            mid_points.append(mid)
            corners.append(corner)
            crops.append(inner)
            angles.append(angle)
            cv2.drawContours(display, [box], -1, (0, 255, 0), 3)
            cv2.circle(display, (int(mid[0]),int(mid[1])), radius=5, color=(255, 0, 0), thickness=10)
            cv2.circle(display, (int(l[0]),int(l[1])), radius=5, color=(255, 0, 0), thickness=3)
            cv2.circle(display, (int(r[0]),int(r[1])), radius=5, color=(0, 0, 255), thickness=3)
            cv2.imwrite("./results/" + name + f"/cropped/crop_{i:02d}.jpg", cropped)
            cv2.imwrite("./results/" + name + f"/cropped/crop_inner{i:02d}.jpg", inner)
            i+=1
    cv2.imwrite("./results/" + name + "/display.jpg", display)
    return crop_pieces, mid_points, corners, crops, angles

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
    
    up_left = r.dot(up_left) + mid
    up_right = r.dot(up_right) + mid
    down_left = r.dot(down_left) + mid
    down_right = r.dot(down_right) + mid
    
    cnt_crop = np.array([up_left, up_right, down_left, down_right], dtype=np.int32)
    rect_crop = cv2.minAreaRect(cnt_crop)
    box_crop = cv2.boxPoints(rect_crop)
    cropped = removeShadow(crop(img, rect_crop, box_crop))
    
    corner = [up_left, up_right, down_left, down_right]
    mid_point = (up_left+up_right+down_left+down_right)//4

    angle = math.atan(abs(down_left[0]-down_right[0]) / abs(down_left[1]-down_right[1])) *180/math.pi
    print(down_left, down_right, angle)
    
    return mid_point, corner, cropped, angle, down_left, down_right

def detect_corners(img, numCorners=4):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, numCorners,0.1,40)
    corners = np.int0(corners)
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
