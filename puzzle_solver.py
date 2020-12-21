import cv2
import numpy as np
import argparse

from Puzzle import PuzzleSolver

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_img", "-i", help="input image path", default = "./images/test/2_original.jpg")
	parser.add_argument("--ori_img", "-o", help="original image path")
	args = parser.parse_args()

	puzzle_solver = PuzzleSolver(args.ori_img, args.input_img)
	puzzle_solver.solve()
