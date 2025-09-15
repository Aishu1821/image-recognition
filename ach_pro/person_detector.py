import cv2
import os
import glob
import pandas as pd

class PersonDetector:
    def __init__(self, videos_path:list[str], photo_path:str):

        print(f'loaded {len(videos_path)} video file')

        self.videos_path = videos_path
        
        # Load the photo named person.jpg
        self.photo = cv2.imread(photo_path)

        print(f'loaded photo from {photo_path}')

        self.sift = cv2.SIFT_create()
        # Create BFMatcher object
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def detect_person(self):
        # Load the Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(r'C:\Users\AISHWARYA\AppData\Local\Programs\Python\Python310\lib\site-packages\cv2\data\haarcascade_frontalface_alt_tree.xml')
        output = []
        # Loop through each video
        for video_path in self.videos_path:
            # Create a video capture object
            cap = cv2.VideoCapture(video_path)

            fps = cap.get(cv2.CAP_PROP_FPS) 
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_in_secs = round(total_frames/fps)
            print(f'checking {video_path} total frames:{total_frames}  video duration:{duration_in_secs} seconds')
            currrent_frame_count=0
            # Loop through each frame in the video
            while cap.isOpened():

                # Read a single frame from the video
                ret, frame = cap.read()

                # Check if we've reached the end of the video
                if not ret:
                    print('completed')
                    break

                print(f'checking frame:{currrent_frame_count}/{total_frames} duration:{currrent_frame_count/fps:.2f}/{duration_in_secs} seconds', end="\r")                
                currrent_frame_count+=1

                # Detect the person in the current frame using the face cascade
                # Store the coordinates of the bounding box around the person
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in faces:
                    face_image = frame[y:y+h, x:x+w]
                    # Calculate the time in seconds for the current frame
                    time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

                    perc = self.get_face_match_percentage(self.photo, face_image)
                    if  perc>11.0:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        print(f"Person present at {time_sec:.2f} seconds in {video_path} percentage_match:{perc}%")
                        output.append({'File': video_path, 'Time (s)': f'{time_sec:.2f}', 'match_percentage':f'{perc:.2f}%'})
                    
                cv2.imshow('rame', frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            # Release the video capture object
            cap.release()
        return output   

    
    def get_descriptors(self, img):
        gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints1, descriptors1 = self.sift.detectAndCompute(gray1, None)
        return descriptors1

    def get_face_match_percentage(self, img1, img2):

        descriptors1 = self.get_descriptors(img1)
        descriptors2 = self.get_descriptors(img2)

        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Compute match percentage
        match_percentage = len(good_matches) / len(matches) * 100
        return match_percentage
    

if __name__=='__main__':

    videos_folder_name = "videos"
    # Load all path of .mp4 videos from the folder
    path_list_of_all_videos = [os.path.join(videos_folder_name, video) for video in os.listdir(videos_folder_name) if video.endswith(".mp4")]
    
    # Load the path photo
    photo_path = os.path.join("person.jpg")
    person_detector = PersonDetector(path_list_of_all_videos, photo_path)
    output = person_detector.detect_person()

    if output:
        df = pd.DataFrame(output)
        df.to_excel('output.xlsx', index=False)
        os.startfile('output.xlsx')
    else:
        print('No instances of the person found in the videos.')

