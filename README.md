# Gesture Controlled Painter 

A real-time gesture-based drawing application built using MediaPipe and OpenCV.
Draw and erase on the screen using finger distance gestures.

---

##  Features
- Draw using index finger gesture
- Erase using pinch gesture
- User-selected pen color
- Persistent drawing canvas
- Real-time webcam interaction

---

##  How It Works
- MediaPipe detects hand landmarks from the webcam
- Distance between thumb and index finger decides the mode:
  - Large distance → Pen
  - Small distance → Eraser
- Drawing is done on a separate canvas and overlaid on live video

---

##  Tech Stack
- Python
- OpenCV
- MediaPipe
- NumPy

---

##  How to Run

```bash
pip install -r requirements.txt
python main.py
