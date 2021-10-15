
# Mediapipe Mapping:
  NOSE = 0
  LEFT_EYE_INNER = 1
  LEFT_EYE = 2
  LEFT_EYE_OUTER = 3
  RIGHT_EYE_INNER = 4
  RIGHT_EYE = 5
  RIGHT_EYE_OUTER = 6
  LEFT_EAR = 7
  RIGHT_EAR = 8
  MOUTH_LEFT = 9
  MOUTH_RIGHT = 10
  LEFT_SHOULDER = 11
  RIGHT_SHOULDER = 12
  LEFT_ELBOW = 13
  RIGHT_ELBOW = 14
  LEFT_WRIST = 15
  RIGHT_WRIST = 16
  LEFT_PINKY = 17
  RIGHT_PINKY = 18
  LEFT_INDEX = 19
  RIGHT_INDEX = 20
  LEFT_THUMB = 21
  RIGHT_THUMB = 22
  LEFT_HIP = 23
  RIGHT_HIP = 24
  LEFT_KNEE = 25
  RIGHT_KNEE = 26
  LEFT_ANKLE = 27
  RIGHT_ANKLE = 28
  LEFT_HEEL = 29
  RIGHT_HEEL = 30
  LEFT_FOOT_INDEX = 31
  RIGHT_FOOT_INDEX = 32

# OpenPose Mapping:
    body_mapping = {'0':  "Nose",      -> 0
                    '1':  "Neck",      -? 11+12

                    '2':  "RShoulder", -> 12
                    '3':  "RElbow",    -> 14
                    '4':  "RWrist",    -> 16

                    '5':  "LShoulder", -> 11
                    '6':  "LElbow",    -> 13
                    '7':  "LWrist",    -> 15

                    '8':  "MidHip"      -> 23+24}

```
obtain_LShoulderPitchRoll_angles(self, P1, P5, P6, P8):
obtain_RShoulderPitchRoll_angles(self, P1, P2, P3, P8):

obtain_LElbowYawRoll_angle(self, P1, P5, P6, P7):
obtain_RElbowYawRoll_angle(self, P1, P2, P3, P4):

obtain_HipPitch_angles(self, P0_curr, P8_curr):

obtain_LShoulderPitchRoll_angles(self, p[11]+p[12], p[11], p[13], p[23+24]):
obtain_RShoulderPitchRoll_angles(self, p[11]+p[12], p[12], p[14], p[23+24]):


obtain_LElbowYawRoll_angle(p[11]+p[12], p[11], p[13], p[15])
obtain_RElbowYawRoll_angle(p[11]+p[12], p[12], p[14], p[16])


obtain_HipPitch_angles(self, p[0], p[23+24]):

    #     {0,  "Nose"},
    #     {1,  "Neck"},
    #     {2,  "RShoulder"},
    #     {3,  "RElbow"},
    #     {4,  "RWrist"},
    #     {5,  "LShoulder"},
    #     {6,  "LElbow"},
    #     {7,  "LWrist"},
    #     {8,  "MidHip"},
    #     {9,  "RHip"},

    body_mapping = {'0':  "Nose",      -> 0
                    '1':  "Neck",      -? 11+12

                    '2':  "RShoulder", -> 12
                    '3':  "RElbow",    -> 14
                    '4':  "RWrist",    -> 16

                    '5':  "LShoulder", -> 11
                    '6':  "LElbow",    -> 13
                    '7':  "LWrist",    -> 15

                    '8':  "MidHip"      -> 23+24}

    needed are ten landmarks in 3D:
      0:
      1:
      2:
```