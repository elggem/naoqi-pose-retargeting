# naoqi-pose-retargeting

![Retargeting Animation](https://raw.githubusercontent.com/elggem/naoqi-pose-retargeting/main/images/animation.gif)

This repository provides scripts to capture 3D human pose using [Mediapipe Pose](https://google.github.io/mediapipe/solutions/pose.html) and retarget it onto Pepper and Nao robots using the NAOqi SDK. It works alongside the receiver code from [this repository](https://github.com/FraPorta/pepper_openpose_teleoperation/tree/main/pepper_teleoperation).

## Installation and Usage

To install, use a Python 3.8 environment and do 

```
pip install -r requirements.txt
```

To analyze a video and visualize the resulting joint angles, do:

```
python teleop.py --video VIDEOFILE --fps 10 --plot_angle_trace
```

To teleoperate Pepper from video, start `pepper_gui.py` from [here](https://github.com/FraPorta/pepper_openpose_teleoperation/tree/main/pepper_teleoperation) (within Python 2.7), and then execute

```
python teleop.py --video VIDEOFILE --fps 10 --enable_teleop
```

To teleoperate Pepper from webcam, use

```
python teleop.py --enable_teleop
```

> :warning: **The Hip Pitch is currently not being calculated correctly**: To fix the hip, change [this line](https://github.com/FraPorta/pepper_openpose_teleoperation/blob/11d4bbd98270fab7a822a7e5b6bbb124f9b6933f/pepper_teleoperation/pepper_approach_control_thread.py#L437) from `self.HipPitch` to `0`. 

## Notes

Mediapipe Landmark mapping ([source](https://google.github.io/mediapipe/solutions/pose.html)).
![Mediapipe Landmark Mapping](https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png)

OpenPose to Mediapipe Body Mapping
```
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

## Todo

 - [ ] Fix hip movement
 - [ ] Write qiBullet simulation part

