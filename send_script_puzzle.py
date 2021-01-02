#!/usr/bin/env python3
import sys
import numpy as np
import math
import cv2
import argparse
from time import sleep
import serial
import os

from mycalibrate import calibrate
from arduino_sucker import Arduino_Sucker
from Puzzle import PuzzleSolver

sys.path.append('/home/robotics/catkin_ws/devel/lib/python2.7/dist-packages')
import rospy
from tm_msgs.msg import *
from tm_msgs.srv import *
    
def send_script(script):
    rospy.wait_for_service('/tm_driver/send_script')
    try:
        script_service = rospy.ServiceProxy('/tm_driver/send_script', SendScript)
        move_cmd = SendScriptRequest()
        move_cmd.script = script
        resp1 = script_service(move_cmd)
    except rospy.ServiceException as e:
        print("Send script service call failed: %s"%e)
    
def set_io(state):
    rospy.wait_for_service('/tm_driver/set_io')
    try:
        io_service = rospy.ServiceProxy('/tm_driver/set_io', SetIO)
        io_cmd = SetIORequest()
        io_cmd.module = 1
        io_cmd.type = 1
        io_cmd.pin = 0
        io_cmd.state = state
        resp1 = io_service(io_cmd)
    except rospy.ServiceException as e:
        print("IO service call failed: %s"%e)

def set_waiting_mission(process_number):
    set_event(SetEventRequest.TAG, process_number, 0)

def wait_for_mission_complete(process_number):
    while True:
        rospy.sleep(0.2)
        res = ask_sta('01', str(process_number), 1)
        data = res.subdata.split(',')
        isComplete = data[1]
        if isComplete == "true":
            print("Waiting Complete.")
            break

def move_arm(X, Y, height, angle, speed):
    targetP1 = str(X) + " , " + str(Y) + " , " + str(height) + " , " + "180.00 , 0.00 , " + str(angle)
    script = "PTP(\"CPP\","+targetP1+","+ str(speed) +",200,0,false)"
    send_script(script)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ori_img", "-o", help="original image path")
    args = parser.parse_args()

    try:
        ############## Connect to Arduino, Webcam, and Robot arm ####################

        # Arduino
        my_sucker = Arduino_Sucker()
        my_sucker.connect()

        # # Ros
        rospy.init_node('send_scripts', anonymous=True)
        set_event = rospy.ServiceProxy('tm_driver/set_event', SetEvent)
        ask_sta = rospy.ServiceProxy('tm_driver/ask_sta', AskSta)

        # Camera
        cap = cv2.VideoCapture(0)
        if not (cap.isOpened()):
            print('Could not open video device')
        else:
            print('Vedio device opened')
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 76)
        print(cap.get(cv2.CAP_PROP_BRIGHTNESS))
        sleep(2)
        cap.set(cv2.CAP_PROP_EXPOSURE,-6)
        print('Camera Parameters')
        for i in range(18):
            print(i,cap.get(i))
        '''
        0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
        1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
        2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
        3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
        4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
        5. CV_CAP_PROP_FPS Frame rate.
        6. CV_CAP_PROP_FOURCC 4-character code of codec.
        7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
        8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
        9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
        10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
        11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
        12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
        13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
        14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
        15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
        16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
        17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
        18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
        '''
        ############## Show the image real-time ################
        while(True):
            ret,frame = cap.read()
            x = 206
            y = 253
            frame = cv2.circle(frame,(x,y),radius = 3,color=(0,0,255),thickness=-1)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('d'):
                break
        
        cv2.destroyAllWindows()
        ############################################### Camera Calibration ###############################################
        intrinsic_matrix = np.loadtxt("./calibration_params/intrinsic_matrix.txt")
        dist_coeff = None
        object_points = np.loadtxt("./calibration_params/object_points.txt")
        image_points = np.loadtxt("./calibration_params/image_points.txt")

        calibration = calibrate(image_points, object_points, intrinsic_matrix, dist_coeff)

        # Use given set of points to verify the calibration.
        print('\n============Verifying the calibration=====================')
        pixel_point = np.array([250.888,106.918]) 
        world_point = np.array([500,200,0])

        # Transform the given world point into pixel point, and see if it match the given pixel point.
        guess_pixel_point = calibration.transform_world_to_pixel(world_point)
        print('Original pixel point : ',pixel_point)
        print('Calculated pixel point by a given world_point: ',guess_pixel_point)

        print('--------------------------\n')

        # Transform the given pixel point into world point, and see if it match the given pixel point.
        guess_world_point = calibration.transform_pixel_to_world(pixel_point)
        print('Original world point : ',world_point)
        print('Calculated world point by a given pixel_point: ',guess_world_point)
        print('================Verify end==========================\n')

        ############################################### Set the releasing place (target_robot_position_list) in advance ###############################################
        target_robot_position_list = np.zeros([4,3,2])
        Rot = np.array([
                        [math.cos(45*math.pi/180), -math.sin(45*math.pi/180), 0, 0],
                        [math.sin(45*math.pi/180), math.cos(45*math.pi/180), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1] 
                        ])
        Trans = np.array([
                         [1, 0, 0, 630],
                         [0, 1, 0, -150],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]
                        ])
        Transform = np.dot(Trans, Rot)
        for i in range(4):
            for j in range(3):
                row = i
                col = j
                target_world_position = np.array([
                                                [30.6736 + i*61.6373],
                                                [26.1407 + j*52.5376],                                                
                                                [0],
                                                [1]
                                                ])
                target_robot_position = np.dot(Transform,target_world_position)
                target_robot_position_list[row][col][0] = target_robot_position[0]
                target_robot_position_list[row][col][1] = target_robot_position[1]
                print(target_world_position, i, j, target_robot_position)

        ############################################ Solving puzzle #############################################
        
        ret, frame = cap.read()

        # frame = cv2.imread('./images/test/lichen_4.jpg')
        # frame = cv2.imread("tmp.jpg")
        # save the original image
        cv2.imwrite('tmp.jpg',frame)

        # ori = cv2.imread(args.input_img)
        # ori = cv2.imread(args.ori_img)
        ori = cv2.imread("lichen_ref1.jpg")



        img = cv2.imread('tmp.jpg')
        # img = cv2.imread('./images/test/lichen_4.jpg')
        # img = cv2.imread('tmp.jpg')
        name = 'tmp'
        # if not os.path.isdir("./results/" + name):
        #     print("creating folder './results/" + name + "'")
        #     os.mkdir("./results/" + name)
        # os.mkdir("./results/" + name + "/cropped")
        puzzle_solver = PuzzleSolver(ori, img, name)
        puzzle_solver.detect_pieces()
        puzzle_solver.solve()
        puzzle_solver.save_result("./results/" + name + "/info.txt")

        puzzle_result = []
        for k, puzzle in enumerate(puzzle_solver.pieces):
            xc = puzzle.pos[0]
            yc = puzzle.pos[1]
            phi = puzzle.orientation
            i = puzzle.target[0]
            j = puzzle.target[1]
            puzzle_result.append([xc, yc, phi, i, j])

        print("\nPuzzle result list: ", puzzle_result)
        print("\nPuzzle: [centroid_points (pixels), orientation_angle (degrees), row, col]")
        for puzzle in puzzle_result:
            print("Puzzle: ", puzzle)

        # Reorder the result, so the arm will suck them in the specified order.
        myorder0 = [0,0,0,1,1,1,2,2,2,3,3,3]
        myorder1 = [0,1,2,0,1,2,0,1,2,0,1,2]
        new_order_result = []
        for order0, order1 in zip(myorder0, myorder1):
            for puzzle in puzzle_result:
                row = puzzle[3]
                col = puzzle[4]
                if(row == order0 and col == order1):
                    new_order_result.append(puzzle)
        print("\nNew order puzzle result list: ", new_order_result)
        print("\nPuzzle: [centroid_points (pixels), orientation_angle (degrees), row, col]")
        for puzzle in new_order_result:
            print("Puzzle: ", puzzle)

        ########################################## Robot moving ################################################
        for index, puzzle in enumerate(new_order_result):
            w, h = frame[:, :, 0].shape
            xc = puzzle[1] + 0.15*w
            yc = puzzle[0] + 0.085*h
            # xc = puzzle[0] 
            # yc = puzzle[1]
            # xc = puzzle[0] 
            # # yc = puzzle[1]
            # while(True):
            #     # ret,frame = cap.read()
            #     display = cv2.imread('./results/tmp/display.jpg')
            #     # frame = cv2.circle(frame,(int(yc),int(xc)),radius = 3,color=(0,255,255),thickness=-1)
            #     # cv2.imshow('frame',frame)
            #     display = cv2.circle(display,(int(yc),int(yc)),radius = 3,color=(0,255,255),thickness=-1)
            #     cv2.imshow('frame',display)
            #     if cv2.waitKey(1) & 0xFF == ord('d'):
            #         break
            # cv2.destroyAllWindows()
            phi = puzzle[2]
            row = puzzle[3]
            col = puzzle[4]
            # From image
            suck_position = calibration.transform_pixel_to_world(np.array([xc, yc]))
            # We set the target to place in advance.
            release_position = target_robot_position_list[row][col]

            suck_x = suck_position[0]+1.4
            suck_y = suck_position[1]-0.8
            release_x = release_position[0]
            release_y = release_position[1]

            height = 220
            angle = phi - 45
            speed = 100  # 100% of 5% (on the screen)

            OK_move = input("\n%d. Ready to go to suck a new puzzle (%d, %d) ? y/n: " %  ((index+1),row,col) )
            if OK_move == 'y':
                
                # Start sucking.
                my_sucker.suck()

                # Move to (xc, yc) of the puzzle.
                height = 192
                angle = phi - 45 # Set the orientation.
                move_arm(suck_x, suck_y, height, angle, speed)
                print("\n1. Move to that puzzle.")

                # Going down.
                height = 186 # sucker will kiss the puzzle at 188(142), and kiss the table at 185.
                move_arm(suck_x, suck_y, height, angle, speed)
                print("2. Going Down.")

                # Finished sucking, move up. 
                height = 220
                move_arm(suck_x, suck_y, height, angle, speed)
                print("3. Finished sucking, move up. ")

                # Move to release place.
                height = 188
                angle = -45
                move_arm(release_x, release_y, height, angle, speed)
                print("4.1 Move to releasing place.")

                # Move to release place.
                height = 142 + 10
                move_arm(release_x, release_y, height, angle, speed)
                print("4.2 Move to releasing place.")

                # Slowly down to fit in. (by drawing a square)               
                print("5. Slowly down to fit in. (by drawing a square)")
                # tunex = [-1, 0, 1, 0, -1, 0, 1, 0, -1, 0]
                # tuney = [0 , 1, 0, -1, 0, 1, 0, -1, 0, 1]
                tunex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                tuney = [0 , 0, 0, 0, 0, 0, 0, 0, 0, 0]
                for i in range(len(tunex)):
                    height -= 1
                    release_x = release_x + tunex[i]
                    release_y = release_y + tuney[i]
                    move_arm(release_x, release_y, height, angle, speed)

                move_arm(release_x, release_y, height, angle, speed)
                # Wait for the last move to complete.
                set_waiting_mission(1) # Set the latest mission to candidates, numbered mission "1".
                wait_for_mission_complete(1) # Wait for the mission numbered "1" to complete.
                sleep(1)
                # Release it.
                my_sucker.release()

                # Finishing this puzzle, move up.
                height = 220
                move_arm(release_x, release_y, height, angle, speed)
                print("Finishing this puzzle, move up.")

                # Always be sucking.
                sleep(3)
                my_sucker.suck()
                
            else:
                print("You rejected that puzzle.")

        
        cap.release()

    except rospy.ROSInterruptException:
        pass