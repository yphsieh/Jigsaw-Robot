from detect_pieces import detect_pieces

import sys
import cv2

thres = [10, 50, 70]
img = cv2.imread(sys.argv[1])
detect_pieces(img, "test", thres)