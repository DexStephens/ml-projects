1. Setup Pi with SSH (Done)
2. Create Script that will check for motion detection on dummy test videos, reverse engineer opencv.absdiff() (Done — see `synthetic_motion_demo.py`)
3. Connect script to real-time camera feed
4. Create and host API that will hold object detection model and handle real-time video feed
5. Write motion detection to hit API to begin streaming video for model analysis of object detection
6. Create model for object detection from scratch to learn training process
7. Create transfer learning model off of ??? model and compare
8. Stream results to server for web interaction to view real-time feed

Good way to think about it:
Physics layer (motion / pixels)
↓
Perception layer (object detection)
↓
Interpretation layer (rules / logic)
↓
Action (alerts, recording, automation)
