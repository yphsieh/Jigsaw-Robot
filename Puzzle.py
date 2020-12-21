import cv2
import sys
import scipy
import numpy as np
import math

from detect_pieces import *
from utility import *

class PuzzleSolver():
	def __init__( self, ori_path, img_path=None ):

		img = image_preprocess(img_path)
		# img = cv2.imread(img_path)
		pieces = detect_pieces(img, 'tmp', thres = [5, 90, 60])
		original = cv2.imread(ori_path, 0)
		self.original = removeShadow(original)

		puzzles = []
		w = 999999
		h = 999999
		for i in range(len(pieces)):
			puzzles.append(Puzzle(pieces[i]))
			tmp_w = puzzles[i].img.shape[0] if puzzles[i].img.shape[0] > puzzles[i].img.shape[1] else puzzles[i].img.shape[1]
			tmp_h = puzzles[i].img.shape[1] if puzzles[i].img.shape[0] > puzzles[i].img.shape[1] else puzzles[i].img.shape[0]
			w = min(w, tmp_w)
			h = min(h, tmp_h)
		self.pieces = puzzles

		self.original = cv2.resize(self.original, (h*3, w*4), interpolation=cv2.INTER_CUBIC)
		cv2.imwrite('images/tmp/resize.png', self.original)

	def solve(self):

		methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
		display = self.original.copy()

		for i in range(len(self.pieces)):
			gray = cv2.cvtColor(self.pieces[i].img, cv2.COLOR_BGR2GRAY)
			gray = getRect(gray)

			w, h = gray.shape[::-1]
			score = -1
			phi_idx = -1
			topleft_idx = -1

			rot = gray.copy()
			for phi in [0, 90, 180, 270]:
				rot = scipy.ndimage.rotate(gray, phi)

				method = eval(methods[3])

				# Apply template Matching
				res = cv2.matchTemplate(self.original, rot, method)
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
			cv2.rectangle(display, top_left, bottom_right, 255, 2)
			cv2.circle(display, (int(top_left[0] + w/2), int(top_left[1] + h/2)), 3, 255, 2)
			cv2.imwrite('images/tmp/matched.jpg', display)

			self.pieces[i].orientation = phi_idx + self.pieces[i].orientation
			self.pieces[i].target = [math.floor((top_left[1] + w/2)/self.original.shape[0] * 4), math.floor((top_left[0] + h/2)/self.original.shape[1] * 3)]
			print(f'angle: {self.pieces[i].orientation}\ttarget: {self.pieces[i].target}')

class Puzzle():
	def __init__(self, piece):
		self.img = piece
		self.orientation = 0
		self.pos = [0,0]		# current position (from image) 
		self.target = [0,0]		# target position (row, column)
		# self.corners = detect_corners(self.img, 4)

	def detect_corners(self): 	# detect and return the corners as a list
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		gray = np.float32(gray)
		dst = cv2.cornerHarris(gray,5,3,0.04)
		ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
		dst = np.uint8(dst)
		ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
		corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
		# for i in range(1, len(corners)): print(corners[i])
		display = self.img
		display[dst>0.1*dst.max()]=[0,0,255]
		cv2.imwrite('images/tmp/corners.jpg', display)
		return corners
