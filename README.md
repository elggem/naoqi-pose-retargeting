# naoqi-pose-retargeting

This repository provides scripts to capture 3D human pose using [Mediapipe Pose](https://google.github.io/mediapipe/solutions/pose.html) and retarget it onto Pepper and Nao robots using the NAOqi SDK.

## Installation and Usage

TBD

## Todo
 - [~] Code cleanup 
 - [~] Actuator velocity and FPS parameters.

 - [ ] Record testcase video that triggers faulty behaviour
 - [ ] Prohibit moving hands behind body to avoid safety artifacts

 - [ ] Try simulation and observe safety features and behaviour
 - [ ] Turning off safety features

 - [ ] Hand jaw movement fixes


## Notes

Mediapipe Landmark mapping ([source](https://google.github.io/mediapipe/solutions/pose.html)).
![Mediapipe Landmark Mapping](https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png)

OpenPose to Mediapipe Body Mapping
````
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
