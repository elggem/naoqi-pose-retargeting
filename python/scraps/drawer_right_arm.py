import time
import almath
import argparse
from naoqi import ALProxy
import random
import decimal



def main():

    robotIP = "192.168.0.141" 
    PORT = 9559
    tts = ALProxy("ALTextToSpeech", robotIP, PORT)
    tts.setLanguage("English")


    rand_yaw = float(decimal.Decimal(random.randrange(-118, 118))/100)
    #rand_pitch = float(decimal.Decimal(random.randrange(-38, 35))/100)

    motionProxy = ALProxy("ALMotion", robotIP, PORT)

    motionProxy.setStiffnesses("RElbowYaw", 1.0)
    motionProxy.setStiffnesses("RShoulderPitch", 1.0)
    motionProxy.setStiffnesses("RWristYaw", 1.0)

    elbow_yaw = "RElbowYaw"
    sholder_pitch = "RShoulderPitch"
    sholder_roll = "RShoulderRoll"
    
    wrist_yaw = "RWristYaw"
    elbow_roll = 'RElbowRoll'

    h_yaw, h_pitch = "HeadYaw", "HeadPitch"
    yaw_angle = -25.0
    pitch_angle = 10.0

    hpitch_angle = pitch_angle * almath.TO_RAD
    hyaw_angle =  yaw_angle * almath.TO_RAD

    sholder_p = 40.0 #90 #40.0
    sholder_r = -40.0
    sholder_p = sholder_p * almath.TO_RAD
    sholder_r = sholder_r * almath.TO_RAD

    fractionMaxSpeed = 0.1
    motionProxy.setAngles(sholder_pitch, sholder_p, fractionMaxSpeed)
    
    motionProxy.setAngles(sholder_roll, sholder_r, fractionMaxSpeed)

    motionProxy.setAngles(elbow_roll, -15* almath.TO_RAD, fractionMaxSpeed)


    

    # head movement

    motionProxy.setAngles("HeadYaw",  hyaw_angle, fractionMaxSpeed)
    motionProxy.setAngles("HeadPitch",  hpitch_angle, fractionMaxSpeed)




    user_msg = "The pen, is in the drawer."

    print "Robot speaks: ", user_msg

    tts.say(str(user_msg))


if __name__ == "__main__":

    main()   

    