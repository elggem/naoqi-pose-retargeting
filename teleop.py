#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Attribution: Part of this code is based on 
https://github.com/Kazuhito00/mediapipe-python-sample/blob/main/sample_pose.py
(Apache 2.0 Licensed)
"""

import copy
import argparse
import cv2 as cv
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

from time import time, sleep

from utils import CvFpsCalc
from utils import KeypointsToAngles
from utils import SocketSend
from utils import SocketReceiveSignal
from utils import calc_bounding_rect, draw_landmarks, plot_world_landmarks, draw_bounding_rect

keypointsToAngles = KeypointsToAngles()

angle_trace = []

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--fps", type=int, default=10)

    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=1)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument('--use_brect', action='store_true')
    parser.add_argument('--plot_world_landmark', action='store_true')
    parser.add_argument('--plot_angle_trace', action='store_true')
    
    parser.add_argument('--enable_teleop', action='store_true')

    args = parser.parse_args()

    return args


def main():
    global angle_trace

    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect
    plot_world_landmark = args.plot_world_landmark
    plot_angle_trace = args.plot_angle_trace
    enable_teleop = args.enable_teleop

    video = args.video
    fps = args.fps

    if len(video) == 0:
        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    else:
        cap = cv.VideoCapture(video)


    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    if enable_teleop:
        # Initialize socket to send keypoints
        ss = SocketSend()
        sr = SocketReceiveSignal()

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    if plot_world_landmark:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

    while True:
        start = time()

        display_fps = cvFpsCalc.get()

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks is not None:
            brect = calc_bounding_rect(debug_image, results.pose_landmarks)
            debug_image = draw_landmarks(debug_image, results.pose_landmarks)
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)

        if plot_world_landmark:
            if results.pose_world_landmarks is not None:
                plot_world_landmarks(plt, ax, results.pose_world_landmarks)

        if results.pose_world_landmarks is not None:
            cv.putText(debug_image, "TRACKING", (cap_width-100, 30),
                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)
            if not do_teleop(results.pose_world_landmarks):
                # limit reached
                cv.putText(debug_image, "LIMIT", (cap_width-100, 90),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)

            if enable_teleop:
                socket_stream_landmarks(ss, results.pose_world_landmarks)

        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 145, 255), 2, cv.LINE_AA)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        cv.imshow('Pepper Teleop', debug_image)

        if (len(video)>0):
            sleep(max(1./fps - (time() - start), 0))

    if plot_angle_trace:
        angle_labels = ["LShoulderPitch","LShoulderRoll", "LElbowYaw", "LElbowRoll", "RShoulderPitch","RShoulderRoll", "RElbowYaw", "RElbowRoll", "HipPitch"]
        angle_trace = np.array(angle_trace)
        lines = plt.plot(angle_trace[:,:])
        plt.legend(iter(lines), angle_labels[:])
        plt.show()

    cap.release()
    cv.destroyAllWindows()

def socket_stream_landmarks(ss, landmarks):
    p = []
    for index, landmark in enumerate(landmarks.landmark):
        p.append([landmark.x, landmark.y, landmark.z])
    # p = np.array(p)

    pNeck =   (0.5 * (np.array(p[11]) + np.array(p[12]))).tolist()
    pMidHip = (0.5 * (np.array(p[23]) + np.array(p[24]))).tolist()
    
    wp_dict = {}

    wp_dict['0'] = p[0]    # Nose
    wp_dict['1'] = pNeck   # Neck
    
    wp_dict['2'] = p[11]   # LShoulder
    wp_dict['3'] = p[13]   # LElbow
    wp_dict['4'] = p[15]   # LWrist
    
    wp_dict['5'] = p[12]   # RShoulder
    wp_dict['6'] = p[14]   # RElbow
    wp_dict['7'] = p[16]   # RWrist
    
    wp_dict['8'] = pMidHip # MidHip

    # print(wp_dict)
    ss.send(wp_dict)

def checkLim(val, limits):
    return val < limits[0] or val > limits[1]


def do_teleop(landmarks):
    global angle_trace
    p = []
    for index, landmark in enumerate(landmarks.landmark):
        p.append([landmark.x, landmark.y, landmark.z])
    p = np.array(p)

    limitsLShoulderPitch = [-2.0857, 2.0857]
    limitsRShoulderPitch = [-2.0857, 2.0857]
    limitsLShoulderRoll  = [ 0.0087, 1.5620]
    limitsRShoulderRoll  = [-1.5620,-0.0087]
    limitsLElbowYaw      = [-2.0857, 2.0857]
    limitsRElbowYaw      = [-2.0857, 2.0857]
    limitsLElbowRoll     = [-1.5620,-0.0087]
    limitsRElbowRoll     = [ 0.0087, 1.5620]
    limitsHipPitch       = [-1.0385, 1.0385]

    pNeck =   (0.5 * (np.array(p[11]) + np.array(p[12]))).tolist()
    pMidHip = (0.5 * (np.array(p[23]) + np.array(p[24]))).tolist()

    LShoulderPitch, LShoulderRoll = keypointsToAngles.obtain_LShoulderPitchRoll_angles(p[11]+p[12], p[11], p[13], p[23]+p[24])
    RShoulderPitch, RShoulderRoll = keypointsToAngles.obtain_RShoulderPitchRoll_angles(p[11]+p[12], p[12], p[14], p[23]+p[24])
    
    LElbowYaw, LElbowRoll = keypointsToAngles.obtain_LElbowYawRoll_angle(pNeck, p[11], p[13], p[15])
    RElbowYaw, RElbowRoll = keypointsToAngles.obtain_RElbowYawRoll_angle(pNeck, p[12], p[14], p[16])

    HipPitch = keypointsToAngles.obtain_HipPitch_angles(pNeck, pMidHip)

    angles = [LShoulderPitch,LShoulderRoll, LElbowYaw, LElbowRoll, RShoulderPitch,RShoulderRoll, RElbowYaw, RElbowRoll, HipPitch]

    print(angles)
    angle_trace.append(angles)

    if (checkLim(LShoulderPitch, limitsLShoulderPitch) or 
        checkLim(RShoulderPitch, limitsRShoulderPitch) or
        checkLim(LShoulderRoll, limitsLShoulderRoll) or 
        checkLim(RShoulderRoll, limitsRShoulderRoll) or
        checkLim(LElbowYaw, limitsLElbowYaw) or 
        checkLim(RElbowYaw, limitsRElbowYaw) or
        checkLim(LElbowRoll, limitsLElbowRoll) or 
        checkLim(RElbowRoll, limitsRElbowRoll)):
        return False
    else:
        return True

if __name__ == '__main__':
    main()
