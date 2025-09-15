import cv2

# Load the video file
video_capture = cv2.VideoCapture('videos/file.mp4')
count=0
# Loop through each frame of the video
while True:
    # Read a single frame from the video
    ret, frame = video_capture.read()

    # If we've reached the end of the video, break out of the loop
    if not ret:
        break

    # Detect faces in the frame
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml')
    faces = face_detector.detectMultiScale(frame)
    # Loop through each face in the frame and save it as an image
    for (x, y, w, h) in faces:
        face_image = frame[y:y+h, x:x+w]
        cv2.imwrite('all_the_faces/face_{}.jpg'.format(count), face_image)
        count+=1
