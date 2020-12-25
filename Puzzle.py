import cv2
import sys
from matplotlib.pyplot import pie
from numpy.lib.function_base import average
import scipy
import numpy as np
import math
import json

from detect_pieces import detect_pieces, image_preprocess, removeShadow

class PuzzleSolver():
    def __init__( self, ori, img, name="test"):
        self.name = name
        # self.camera_img = image_preprocess(img)
        self.camera_img = img
        self.original = removeShadow(ori)
        self.pieces = []
        self.w = 0
        self.h = 0

    def detect_pieces(self):
        imgs = self.camera_img
        pieces, mid_points, corners, crops = detect_pieces(imgs, self.name)
        ws = []
        hs = []
        for i in range(len(pieces)):
            self.pieces.append(Puzzle(pieces[i], mid_points[i], corners[i], crops[i]))
            # print(self.pieces[i].inner.shape)
            tmp_w = max(self.pieces[i].inner.shape[0], self.pieces[i].inner.shape[1])
            tmp_h = min(self.pieces[i].inner.shape[0], self.pieces[i].inner.shape[1])
            ws.append(tmp_w)
            hs.append(tmp_h)
        self.w = average(ws)
        self.h = average(hs)
        self.original = cv2.resize(self.original, (int(self.h*3), int(self.w*4)), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("./results/" + self.name + '/resize.png', self.original)


    def solve(self, methodId=3):
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        display = self.original.copy()

        for idx, piece in enumerate(self.pieces):
            gray = cv2.cvtColor(piece.inner, cv2.COLOR_BGR2GRAY)
            # gray = getRect(gray, piece.corner)
            ori_gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)

            w, h = gray.shape[::-1]
            score = -1
            phi_idx = -1
            topleft_idx = -1

            rot = gray.copy()
            for phi in [0, 90, 180, 270]:
                rot = scipy.ndimage.rotate(gray, phi)

                method = eval(methods[methodId])

                # Apply template Matching
                res = cv2.matchTemplate(ori_gray, rot, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    top_left = min_loc
                else:
                    top_left = max_loc

                if max_val > score :
                    score = max_val
                    phi_idx = phi
                    topleft_idx = top_left

            top_left = topleft_idx
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(display, top_left, bottom_right, (255, 0, 0), 8)
            mid = (int(top_left[0] + w/2), int(top_left[1] + h/2))
            cv2.circle(display, mid, 3, 255, 2)
            cv2.putText(display, f"{idx}", mid, cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 10, cv2.LINE_AA)
            print("\nsaving result at ./results/" + self.name + "/matched.jpg")
            cv2.imwrite("./results/" + self.name + '/matched.jpg', display)

            piece.orientation = phi_idx + piece.orientation
            piece.target = [math.floor((top_left[1] + w/2)/self.original.shape[0] * 4), math.floor((top_left[0] + h/2)/self.original.shape[1] * 3)]
            print(f'angle: {piece.orientation}\ttarget: {piece.target}')
    def save_result(self, path):
        info = dict()
        for idx, p in enumerate(self.pieces):
            info[idx] = {
				"posx":  int(p.middle_point[0]), 
				"posy":  int(p.middle_point[1]), 
				"orientation": p.orientation, 
				"targetx": int(p.target[0]*self.h+self.h/2), 
                "targety": int(p.target[1]*self.w+self.w/2)
			}
        with open(path, 'w') as f:
            json.dump(info, f)

class Puzzle():
    def __init__(self, piece, mid, corner, inner):
        self.img = piece
        self.middle_point = mid
        self.orientation = 0
        self.pos = [0,0]        # current position (from image) 
        self.target = [0,0]        # target position (row, column)
        self.corner = corner
        self.inner = inner