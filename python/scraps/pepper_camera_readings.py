import sys
import numpy as np
import cv2
from naoqi import ALProxy



ip_addr, port_num = "192.168.0.165", 9559

# get NAOqi module proxy
videoDevice = ALProxy('ALVideoDevice', ip_addr, port_num)

# subscribe top camera
AL_kTopCamera = 0
AL_kQVGA = 2           # 320x240
AL_kBGRColorSpace = 13
captureDevice = videoDevice.subscribeCamera(
    "test", AL_kTopCamera, AL_kQVGA, AL_kBGRColorSpace, 10)

# create image
width = 320*2
height = 240*2
image = np.zeros((height, width, 3), np.uint8)

while True:

    # get image
    result = videoDevice.getImageRemote(captureDevice);

    if result == None:
        print 'cannot capture.'
    elif result[6] == None:
        print 'no image data string.'
    else:

        # translate value to mat
        values = map(ord, list(result[6]))
        i = 0
        for y in range(0, height):
            for x in range(0, width):
                image.itemset((y, x, 0), values[i + 0])
                image.itemset((y, x, 1), values[i + 1])
                image.itemset((y, x, 2), values[i + 2])
                i += 3

        # show image
        cv2.imshow("pepper-top-camera-320x240", image)

    # exit by [ESC]
    if cv2.waitKey(33) == 27:
        break