import cv2
import numpy as np
import argparse

from Puzzle import PuzzleSolver

parser = argparse.ArgumentParser()
parser.add_argument("--input_img", "-i", help="input image path", default = "./images/test/2_original.jpg")
parser.add_argument("--ori_img", "-o", help="original image path")
args = parser.parse_args()

if __name__=="__main__":
	img = cv2.imread(args.input_img)
	ori = cv2.imread(args.ori_img)
	# ori = cv2.resize(ori, (1291, 10211), interpolation=cv2.INTER_LINEAR)
	name = args.input_img.split('/')[-1].split('.')[0]
	puzzle_solver = PuzzleSolver(ori, img, name)
	puzzle_solver.main()
	puzzle_solver.solve()
