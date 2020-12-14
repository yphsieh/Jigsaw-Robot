import cv2
import sys
import scipy

from detect_pieces import *

class PuzzleSolver():
	def __init__(self, img=None):
		self.original_img = img

		pieces = detect_pieces(self.original_img)
		puzzles = []
		for i in range(len(pieces)):
			puzzles.append(Puzzle(pieces[i]))
		self.pieces = puzzles

	# def solve(self):
		

class Puzzle():
	def __init__(self, piece):
		self.img = piece
		self.corners = self.detect_corners()
		# self.orientaion = self.cal_orientation()
		self.pos = [0,0] # current position --> target position

	def detect_corners(self):
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		gray = np.float32(gray)
		dst = cv2.cornerHarris(gray,5,3,0.04)
		ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
		dst = np.uint8(dst)
		ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
		corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
		for i in range(1, len(corners)): print(corners[i])
		display = self.img
		display[dst>0.1*dst.max()]=[0,0,255]
		cv2.imwrite('images/tmp/corners.jpg', display)
		return corners

	# def cal_orientation(self):

	def get_corners(self):
		return self.corners

	def get_pos(self):
		return self.pos

	def get_orientation(self):
		return self.orientaion
