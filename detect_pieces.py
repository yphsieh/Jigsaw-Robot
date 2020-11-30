import matplotlib.pyplot as plt
import numpy as np
import cv2
import random as rd

def detect_pieces(im):
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    med_h = np.median(im_hsv[:, :, 0])
    med_s = np.median(im_hsv[:, :, 1])
    med_v = np.median(im_hsv[:, :, 2])

    lower_thres = np.array([med_h-5,med_s-80,med_v-80])
    upper_thres = np.array([med_h+5,med_s+80,med_v+80])

    masked_img = cv2.inRange(im_hsv, lower_thres, upper_thres)
    res_img = cv2.bitwise_and(im, im, mask = (255-masked_img))

    kernel = np.ones((3,3), np.uint8)
    masked_img = cv2.erode(masked_img, kernel, iterations=2)
    masked_img = cv2.dilate(masked_img, kernel, iterations=2)

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
            cv2.imwrite(f"./images/test/cropped/crop_{i:02d}.jpg", cropped)
    cv2.imwrite("display.jpg", display)
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
    im = cv2.imread("images/puzzle_images/all/all_001.jpg")
    detect_pieces(im)