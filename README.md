# Image Recognition – IZU

## Description
A Python-based real-time image recognition system that detects and matches a specific person in video files using computer vision techniques. The system uses OpenCV’s Haar cascades for face detection and SIFT feature matching to identify the target person.

## Key Features
- Detects faces in video frames using OpenCV Haar cascades
- Matches faces against a reference image using SIFT feature descriptors
- Outputs the timestamp and match percentage when the person is detected
- Saves results to `output.xlsx` for easy review
- Optionally extracts all faces from videos and saves them as images

## Technologies Used
- Python 3.x
- OpenCV
- pandas
- Excel output for detected instances

## Setup & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/image-recognition.git
