import cv2
import numpy as np
from argparse import ArgumentParser

from detect_pieces import detect_pieces

parser = ArgumentParser()
parser.add_argument("--input_img", "-i", required=True, help="input image path")
args = parser.parse_args()

class PuzzleSolver():
    def __init__(self, img=None):
        self.original_img = img
        self.pieces = []
    
    def main(self):
        self.pieces = detect_pieces(self.original_img)
        print(len(self.pieces))

if __name__=="__main__":
    img = cv2.imread(args.input_img)
    puzzle_solver = PuzzleSolver(img)
    puzzle_solver.main()