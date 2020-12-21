import cv2
import sys
import scipy
import numpy as np

from detect_pieces import detect_pieces, image_preprocess
from match_template import match_template

class PuzzleSolver():
	def __init__(self, ori, img, name="test"):
		self.original_img = ori
		self.camera_img = img
		self.name = name
		self.pieces = []

	def main(self):
		imgs = image_preprocess(self.camera_img)
		pieces = detect_pieces(imgs, self.name)
		# print(pieces)
		puzzles = []
		for p in pieces:
			self.pieces.append(p)
			# match = match_template(self.original_img, p)

		# print(self.pieces[0].get_corners().shape)

	# def solve(self):
		

class Puzzle():
	def __init__(self, piece):
		self.img = piece
		self.corners = self.detect_corners()
		self.orientaion = 0 # self.cal_orientation()
		self.pos = [0,0]	# current position
		self.target = [0,0]	# target position

	def detect_corners(self):
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

	# def cal_orientation(self):

	def get_corners(self):
		return self.corners

	def get_pos(self):
		return self.pos

	def get_target(self):
		return self.target

	def get_orientation(self):
		return self.orientaion
