# 🛡️ Dr. Strange Shields - Gesture Control System

**Created by: Soumya Suman Kar**

A real-time hand gesture recognition system that creates magical shield effects inspired by Doctor Strange, using computer vision and machine learning.


![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Gesture Sequence](#-gesture-sequence)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **Real-time Hand Gesture Recognition**: Uses MediaPipe Holistic for accurate hand tracking
- **Machine Learning Classification**: SVM model for gesture classification
- **Visual Effects**: Overlays magical shield effects on detected hands
- **Multiple Output Modes**:
  - OpenCV window display
  - Virtual camera output (for use in video calls, streaming, etc.)
  - Both simultaneously
- **Sequential Gesture Activation**: Requires a specific sequence of gestures to activate shields
- **Configurable Parameters**: Adjustable thresholds and confidence levels
- **Graceful Shutdown**: Clean resource management with Ctrl+C handling

## 🎬 Demo

The system detects hand gestures in real-time and overlays shield effects when the correct sequence is performed:

1. Perform gesture KEY_1
2. Within 2 seconds, perform gesture KEY_2
3. Within 2 seconds, perform gesture KEY_3
4. Shields activate! ✨
5. Perform gesture KEY_4 to deactivate

## 🔧 Installation

### Prerequisites

- Python 3.9 or higher
- Webcam or camera device
- (Optional) Virtual camera software if using virtual camera output

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Clone the Repository

```bash
git clone https://github.com/SOUMYA0023/Python-DrStrangeShield-master.git
cd Python-DrStrangeShield-master
```

### Project Structure

The project has the following structure:

```
Python-DrStrangeShield-master/
├── shield.py                  # Main application script
├── utils.py                   # Utility functions for MediaPipe
├── dataset_collection.py      # Script for collecting gesture training data
├── rh_dataset_collection.py   # Right hand dataset collection
├── train_svm.py              # SVM model training script
├── requirements.txt           # Python dependencies
├── models/
│   └── model_svm.sav         # Pre-trained SVM model
├── effects/
│   └── shield.mp4            # Shield video effect
├── data/                      # Directory for training datasets
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Usage

### Basic Usage

Run with default settings (both OpenCV window and virtual camera):

```bash
python shield.py
```

### Advanced Usage

#### OpenCV Window Only

```bash
python shield.py --output window
```

#### Virtual Camera Only (Headless)

```bash
python shield.py --output virtual
```

#### Custom Camera

```bash
python shield.py --camera_id 1
```

#### Full Configuration

```bash
python shield.py \
  --model models/model_svm.sav \
  --threshold 0.9 \
  --det_conf 0.5 \
  --trk_conf 0.5 \
  --camera_id 0 \
  --shield effects/shield.mp4 \
  --output both
```

### Command Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--model` | `-m` | Path to trained ML model file | `models/model_svm.sav` |
| `--threshold` | `-t` | Prediction threshold (0-1) | `0.9` |
| `--det_conf` | `-dc` | Detection confidence (0-1) | `0.5` |
| `--trk_conf` | `-tc` | Tracking confidence (0-1) | `0.5` |
| `--camera_id` | `-c` | Camera device ID | `0` |
| `--shield` | `-s` | Path to shield video effect | `effects/shield.mp4` |
| `--output` | `-o` | Output mode: `window`, `virtual`, or `both` | `both` |

### Helpful Tips

**First time using the system?**
- Run with `--threshold 0.7` for easier gesture detection
- Watch the on-screen "Detected:" text to learn which poses trigger which keys
- The system shows real-time feedback - experiment with different hand positions!
- Audio feedback (beeps) confirm when gestures are successfully recognized

### Controls

- **Q key**: Quit application (when OpenCV window is active)
- **Ctrl+C**: Graceful shutdown from terminal

## 🧠 How It Works

### 1. Hand Detection

The system uses **MediaPipe Holistic** to detect and track hand landmarks in real-time:
- Detects 21 landmarks per hand
- Tracks both left and right hands simultaneously
- Calculates bounding boxes for each detected hand

### 2. Gesture Classification

- Extracts normalized hand landmark coordinates
- Feeds coordinates to a pre-trained **SVM (Support Vector Machine)** model
- Classifies gestures with probability scores
- Requires high confidence (>0.85 by default) for gesture recognition

### 3. Sequential Activation

The shield system requires a specific sequence:

```
KEY_1 → (within 2s) → KEY_2 → (within 2s) → KEY_3 → SHIELDS ACTIVATED
```

This prevents accidental activation and adds a "magical" element to the interaction.

### 4. Visual Effects

When shields are active:
- Reads frames from the shield video effect
- Removes black background (chroma keying)
- Scales and positions shields relative to hand positions
- Blends shield effect with camera feed using alpha blending
- Adjusts shield size based on hand bounding box dimensions (scaled by 1.5x)

### 5. Output Modes

- **Window Mode**: Displays in OpenCV window
- **Virtual Camera**: Outputs to virtual camera device (for OBS, Zoom, Teams, etc.)
- **Both**: Simultaneous output to both

## ⚙️ Configuration

### Adjusting Shield Size

Modify the `scale` variable in the code (default: 1.5):

```python
scale = 1.5  # Increase for larger shields, decrease for smaller
```

### Gesture Timing

Modify timeout values for gesture sequences:

```python
# Current: 2 seconds between gestures
if t1 + timedelta(seconds=2) > t2:  # Change 2 to desired seconds
```

### Prediction Confidence

Adjust the probability threshold for gesture recognition:

```python
if (prediction == 'key_1') and (pred_prob > 0.85):  # Change 0.85 to desired threshold
```

## 📂 Project Structure

### Required Files

#### `utils.py`
Must contain the following functions:
- `mediapipe_detection(frame, model)`: Processes frame with MediaPipe
- `get_center_lh(frame, results)`: Gets left hand bounding box
- `get_center_rh(frame, results)`: Gets right hand bounding box
- `points_detection_hands(results)`: Extracts hand landmarks for ML model

#### `models/model_svm.sav`
Pre-trained SVM model that recognizes:
- `key_1`: First gesture in sequence
- `key_2`: Second gesture in sequence
- `key_3`: Third gesture in sequence
- `key_4`: Deactivation gesture

#### `effects/shield.mp4`
Video file containing the shield effect with black background for transparency.

## 🎯 Gesture Sequence

### Learning the Gestures

**Don't know what gestures to perform?** The system will help you discover them!

1. **Run the application** and show both hands to the camera
2. **Watch the bottom of the screen** - it shows:
   - **"Detected: KEY_X (confidence)"** - What gesture the model currently sees
   - **"Next: Perform KEY_X gesture"** - Which gesture you need to do next
3. **Try different hand poses** while watching the detection feedback
4. When the model detects a gesture with **high confidence (>0.70)**, you'll:
   - Hear a beep sound 🔊
   - See the key status change (🔑1✅)
   - See progress in the terminal

### Gesture Discovery Tips

- **Both hands must be visible** for gesture recognition
- **Try various poses**: open palms, closed fists, fingers pointing, hands together, etc.
- Watch the **"Detected:"** text to see what the model recognizes
- **Green text** = high confidence, **Orange text** = lower confidence
- The system uses the **pre-trained gestures** from the SVM model
- Reference the images below to see example poses

### Activation Sequence

### Activation Sequence

- In order to activate the shields you have to perform a "magical" sequence of hands position.

**The system will guide you:**
- Screen shows: "Next: Perform KEY_1 gesture"
- Once KEY_1 is detected → "Next: Perform KEY_2 gesture (2s left)"
- Once KEY_2 is detected → "Next: Perform KEY_3 gesture (2s left)"  
- Once KEY_3 is detected → **Shields activate!** ✨

<br>
<p align="center">
  <img width="320"  src="./images/position_1.png">
  <img width="320"  src="./images/position_2.png">
  <img width="320"  src="./images/position_3.png">
</p>
<br>

- In order to deactivate the shields you have to execute a "magical" hands position.

<br>
<p align="center">
  <img width="360"  src="./images/position_4.png">
</p>
<br>


### Activation Sequence

1. **KEY_1**: Perform first gesture with both hands visible
   - Status: 🔑1✅ 🔑2❌ 🔑3❌
   
2. **KEY_2**: Within 2 seconds, perform second gesture
   - Status: 🔑1✅ 🔑2✅ 🔑3❌
   
3. **KEY_3**: Within 2 seconds, perform third gesture
   - Status: 🔑1✅ 🔑2✅ 🔑3✅
   - **Shields activate!** 🛡️ ON

### Deactivation

4. **KEY_4**: Perform deactivation gesture with both hands
   - All keys reset
   - Shields deactivate 🛡️ OFF

## 🐛 Troubleshooting

### Camera Not Found

```bash
# Try different camera IDs
python main.py --camera_id 1  # or 2, 3, etc.
```

### Low Performance

- Reduce camera resolution
- Lower MediaPipe model complexity (already set to 0)
- Use `--output window` to disable virtual camera
- Ensure good lighting conditions for better hand detection
- Close other CPU-intensive applications

### Model Not Found

```bash
# Ensure the models directory exists with the trained model
ls models/model_svm.sav
```

If the model file is missing, you'll need to train a new model using the provided training scripts.

### Virtual Camera Not Working

- Install virtual camera driver (e.g., OBS Virtual Camera, v4l2loopback on Linux)
- Check if virtual camera device is available
- Use `--output window` to test without virtual camera

### Gestures Not Recognized

- Ensure both hands are clearly visible
- Check lighting conditions
- Lower threshold: `--threshold 0.7`
- Lower detection confidence: `--det_conf 0.3`

### Shield Effect Not Visible

- Verify `effects/shield.mp4` exists
- Ensure shield video has black background
- Check shield video format (MP4 recommended)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Areas for Improvement

- [ ] Add more gesture types
- [ ] Implement gesture training interface
- [ ] Support custom shield effects
- [ ] Add sound effects
- [ ] Implement gesture recording for model training
- [ ] Add configuration file support
- [ ] Create GUI for parameter adjustment

## 🔧 Advanced Features

### Training Your Own Gestures

If you want to train custom gestures:

1. **Collect Training Data**:
   ```bash
   python dataset_collection.py
   ```
   Follow on-screen instructions to record gesture samples.

2. **Train the Model**:
   ```bash
   python train_svm.py
   ```
   This will create a new `model_svm.sav` file in the `models/` directory.

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.9 or higher
- **Webcam**: Built-in or external USB camera (720p or higher recommended)
- **RAM**: Minimum 4GB (8GB recommended)
- **Processor**: Multi-core processor recommended for smooth real-time processing

### Dependencies Explained

- **opencv-python** & **opencv-contrib-python**: Computer vision and image processing
- **mediapipe**: Google's ML framework for hand/pose detection
- **scikit-learn**: Machine learning library for SVM classifier
- **numpy**: Numerical computations
- **pandas**: Data manipulation for training
- **pyvirtualcam**: Virtual camera output support
- **pygame**: Audio feedback generation
- **matplotlib**: Data visualization during training

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Soumya Suman Kar

## 👨‍💻 Author

**Soumya Suman Kar**
- Sole creator and developer of this project
- All rights reserved under MIT License

## 🙏 Acknowledgments

- **MediaPipe**: Google's ML framework for hand tracking
- **OpenCV**: Computer vision library
- **pyvirtualcam**: Virtual camera library
- **pygame**: Audio synthesis and playback
- **scikit-learn**: Machine learning library
- **Marvel Studios**: Inspiration from Doctor Strange

## 🚨 Important Notes

- **Privacy**: All video processing happens locally on your machine. No data is sent to external servers.
- **Performance**: First run may take a few seconds to initialize MediaPipe models.
- **Camera Access**: Ensure your camera is not being used by another application.
- **Audio**: Audio feedback requires pygame. If audio fails to initialize, the application will continue without sound.
- **Virtual Camera**: For virtual camera output, you may need to install additional drivers:
  - **Windows**: OBS Virtual Camera or similar
  - **macOS**: OBS Virtual Camera
  - **Linux**: v4l2loopback kernel module

## ❓ FAQ

**Q: What gestures should I perform?**
A: The system uses pre-trained gestures. Run the application and experiment with different hand poses while watching the on-screen feedback. The system will show what gesture it detects.

**Q: Why aren't my gestures being recognized?**
A: Ensure both hands are clearly visible, you have good lighting, and you're performing gestures with confidence. Try lowering the threshold with `--threshold 0.7`.

**Q: Can I use this in Zoom/Teams/OBS?**
A: Yes! Use `--output virtual` or `--output both` to send the output to a virtual camera that can be used in video conferencing apps.

**Q: How do I exit the application?**
A: Press 'Q' when the OpenCV window is active, or use Ctrl+C in the terminal.

**Q: Can I customize the shield effect?**
A: Yes! Replace the `effects/shield.mp4` file with your own video effect (ensure it has a black background for transparency).

## 📞 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Made with ✨ magic and 🐍 Python by Soumya Suman Kar**
