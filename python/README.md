# naoqi-pose-retargeting

This repository provides scripts to capture 3D human pose using [Mediapipe Pose](https://google.github.io/mediapipe/solutions/pose.html) and retarget it onto Pepper and Nao robots using the NAOqi SDK.

## Install

Mediapipe needs to be compiled from source to be compatible with Python 2.7 to work alongside the NAOqi SDK. Under Ubuntu, please follow [these instructions](https://google.github.io/mediapipe/getting_started/install.html#installing-on-debian-and-ubuntu) to setup Bazel and OpenCV, then follow [these instructions](https://google.github.io/mediapipe/getting_started/python.html#building-mediapipe-python-package) *in a Python 2.7 environment* (we used Miniconda2) to install the package. After, follow [these instructions](http://doc.aldebaran.com/2-5/dev/python/install_guide.html) to install NAOqi SDK on your Python installation.

To test if everything is correctly setup you can do the following in `python2`:

```
import mediapipe
import naoqi
```

## Architecture

TBD
