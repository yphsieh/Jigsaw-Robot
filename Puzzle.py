import cv2
import sys
import scipy

from detect_pieces import *
from utility import *

class PuzzleSolver():
	def __init__( self, img=None ):
		self.original_img = img

		pieces = detect_pieces(self.original_img, 'tmp')
		puzzles = []
		for i in range(len(pieces)):
			puzzles.append(Puzzle(pieces[i]))
		self.pieces = puzzles

	def solve(self):
		for i in range(len(self.pieces)):
			rotated = scipy.ndimage.rotate(self.pieces[i].img, self.pieces[i].orientation)
			# for 0/90/180/270:
			# 	cv2.matchTemplate(self.original_img, rotated, method)
			# 	if found: self.orientation = 0/90/180/270 + self.orientation
		

class Puzzle():
	def __init__(self, piece):
		self.img = piece
		self.corners = self.detect_corners()
		self.orientation = 0 	# self.cal_orientation() yet to be done
		self.pos = [0,0]		# current position (from image) 
		self.target = [0,0]		# target position (row, column)

	def detect_corners(self): 	# detect and return the corners as a list
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		# gray = np.float32(gray)
		# dst = cv2.cornerHarris(gray,2,3,0.04)
		# ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
		# dst = np.uint8(dst)
		# ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
		# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
		# corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
		# for i in range(1, len(corners)): print(corners[i])
		# display = self.img
		# display[dst>0.1*dst.max()]=[0,0,255]
		# cv2.imwrite('images/tmp/corners.jpg', display)

		# blur_img = cv2.GaussianBlur(gray, (3,3), 0)
		try:
			edges = cv2.Canny(gray, 200, 250, apertureSize = 3)
			lines = cv2.HoughLines(edges,1,np.pi/180,170)

			for i in range(len(lines)):
				rho, theta = lines[i][0][0],lines[i][0][1]
				a = np.cos(theta)
				b = np.sin(theta)
				x0 = a*rho
				y0 = b*rho
				x1 = int(x0 + 1000*(-b))
				y1 = int(y0 + 1000*(a))
				x2 = int(x0 - 1000*(-b))
				y2 = int(y0 - 1000*(a))
				cv2.line(self.img,(x1,y1),(x2,y2),(0,0,255),2)
				# if len(lines) >= 2: cv2.imwrite('images/tmp/hough.jpg', self.img)
			print(len(lines), "lines")

		except:
			print("no line")
		
		# return corners

	# def cal_orientation(self): # calculate and return the orientation
