import cv2
import sys
import scipy

from detect_pieces import *
from utility import *

class PuzzleSolver():
	def __init__( self, img_path=None ):

		img = image_preprocess(img_path)
		# img = cv2.imread(img_path)
		pieces = detect_pieces(img, 'tmp', thres = [5, 90, 60])
		original = cv2.imread('images/test/ref_image.png', 0)
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

		print(w, h)
		self.original = cv2.resize(self.original, (h*3, w*4), interpolation=cv2.INTER_CUBIC)
		cv2.imwrite('images/tmp/resize.png', self.original)
		print(self.original.shape)

	def solve(self):

		methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
		display = self.original.copy()

		for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]: #range(len(self.pieces)):
			# rotated = scipy.ndimage.rotate(self.pieces[i].img, self.pieces[i].orientation)
			gray = cv2.cvtColor(self.pieces[i].img, cv2.COLOR_BGR2GRAY)
			gray = getRect(gray)

			w, h = gray.shape[::-1]
			score = -1
			phi_idx = -1
			topleft_idx = -1

			for phi in [0, 90, 180, 270]:
				gray = scipy.ndimage.rotate(gray, phi)

				method = eval(methods[3])

				# Apply template Matching
				res = cv2.matchTemplate(self.original, gray, method)
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

			self.pieces[i].orientation = phi_idx + self.pieces[i].orientation
			top_left = topleft_idx
			bottom_right = (top_left[0] + w, top_left[1] + h)

			print(phi_idx, max_val, top_left, bottom_right)

			cv2.rectangle(display, top_left, bottom_right, 255, 2)

			cv2.imwrite('images/tmp/matching.jpg', gray)
			cv2.imwrite('images/tmp/matched.jpg', display)
	

class Puzzle():
	def __init__(self, piece):
		self.img = piece
		# self.corners = detect_corners(self.img, 4)
		self.orientation = 0
		self.pos = [0,0]		# current position (from image) 
		self.target = [0,0]		# target position (row, column)

