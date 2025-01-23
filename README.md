# Fall Detection with OpenCV and MediaPipe

This project leverages OpenCV and MediaPipe Pose to implement a real-time fall detection system using a webcam. The system captures and processes video frames to analyze body landmarks and detect falls based on body posture and movement.

## Features

- **Real-time video processing**: Utilizes OpenCV to process video frames in real-time.
- **Pose detection**: Employs MediaPipe Pose to extract 33 key body landmarks.
- **Angle calculation**: Calculates joint angles to analyze posture and detect potential falls.
- **Full-screen video display**: Displays video feed with overlayed landmarks in full-screen mode.
- **Customizable detection threshold**: Configurable time-to-alert duration for fall detection.

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Riddikuluz/falling_detection.git
   cd falling_detection
   ```

2. Install the required dependencies:

   ```bash
   python3 -m venv my_env
   source my_env/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   The `requirements.txt` file should contain:

   ```
   opencv-python
   mediapipe
   numpy
   ```

3. Ensure your webcam is connected and accessible.

## Usage

1. Run the fall detection script:

   ```bash
   python fall_detection.py
   ```

2. The system will open a full-screen video window and start detecting pose landmarks in real-time.

3. If a potential fall is detected, the system will trigger an alert after the configured time threshold.

## Code Highlights

### Pose Detection

The project uses MediaPipe Pose to detect body landmarks:

```python
with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
    results = pose.process(image)
    landmarks = results.pose_landmarks.landmark
```

### Angle Calculation

Joint angles are calculated to analyze posture:

```python
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return int(angle)
```

### Fall Detection Logic

The system identifies potential falls based on landmark positions:

```python
fall_start_time = None
time_2_alert = 10  # seconds
if fall_detected:
    if fall_start_time is None:
        fall_start_time = time.time()
    elif time.time() - fall_start_time > time_2_alert:
        print("Fall detected! Sending alert...")
```

## Configuration

- **Resolution**: Default resolution is set to 1280x720. Modify it in the script:
  ```python
  width = 1280
  height = 720
  ```
- **Detection thresholds**: Adjust `min_detection_confidence` and `min_tracking_confidence` in the MediaPipe Pose setup for better accuracy.

## Troubleshooting

- **Webcam not detected**: Ensure your webcam is properly connected. Check access permissions and test with another application.
- **Low performance**: Reduce resolution or adjust confidence thresholds for better performance on low-end hardware.

## Contributing

Feel free to submit issues or pull requests to enhance this project. Contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/aws/mit-0) file for details.

## Acknowledgments

- [onenationonemind1](https://github.com/onenationonemind1) for the code.
- [MediaPipe](https://mediapipe.dev/) for providing the Pose detection solution.
- [OpenCV](https://opencv.org/) for video processing capabilities.
