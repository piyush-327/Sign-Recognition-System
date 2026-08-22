import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import pickle
import pyttsx3
import threading

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="dataset/hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO)
with HandLandmarker.create_from_options(options) as landmarker:

    cam = cv2.VideoCapture(0)
    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),

        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (0, 9),

        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (0, 13),

        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),

        (17, 18),
        (18, 19),
        (19, 20),
        (0, 17)
    ]

    with open('model1.pkl', "rb") as f:
        model = pickle.load(f)

    with open('model_numbers.pkl', 'rb') as f:
        model_numbers = pickle.load(f)

    def speak_word(text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    current_mode = "Alphabet"
    current_word = ""
    previous_prediction = None
    consecutive_frames = 0
    frame_threshold = 20


    while True:
        success, frame = cam.read()
        if not success:
            print("Failed to Load Camera")
            print("Exiting Program")
            exit()

        frame = cv2.flip(frame,1)
        current_time_ms = int(time.time() * 1000)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,c = frame.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        hand_landmarker_result = landmarker.detect_for_video(mp_image, current_time_ms)

        if hand_landmarker_result.hand_landmarks:
            for hand_idx, hand_landmark in enumerate(hand_landmarker_result.hand_landmarks):
                handedness = hand_landmarker_result.handedness[hand_idx]
                hand_label = handedness[0].category_name
                score = handedness[0].score

            for landmark_id, landmark in enumerate(hand_landmark):
                # Scale normalized coordinates (0.0 to 1.0) to frame pixels
                (pixel_x,pixel_y) = (int(landmark.x * w), int(landmark.y * h))

                cv2.circle(frame, (pixel_x, pixel_y), 6, (0, 255, 0), cv2.FILLED)
            for start_idx, end_idx in connections:
                lm1 = hand_landmark[start_idx]
                lm2 = hand_landmark[end_idx]
                pt1 = (int(lm1.x * w), int(lm1.y * h))
                pt2 = (int(lm2.x * w), int(lm2.y * h))
                cv2.line(frame,pt1,pt2,(255,0,0),2)

            x_wrist = hand_landmark[0].x
            y_wrist = hand_landmark[0].y
            z_wrist = hand_landmark[0].z

            sub = []
            for landmark in hand_landmark:
                x_translated = landmark.x - x_wrist
                y_translated = landmark.y - y_wrist
                z_translated = landmark.z - z_wrist
                sub.append(x_translated)
                sub.append(y_translated)
                sub.append(z_translated)

            M = max(abs(val) for val in sub)
            if M== 0:
                M = 1.0
            normalized_coordinates = []
            for value in sub:
                normalized_coordinates.append(value / M)

            if current_mode == "Alphabet":
                prediction_array = model.predict([normalized_coordinates])
                current_pred = str(prediction_array[0])
                text = "Letter: " + str(current_pred)
                position = (500, 80)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(frame, text, position, font, fontScale, color, thickness, cv2.LINE_AA)

            elif current_mode == "number":
                prediction_array = model_numbers.predict([normalized_coordinates])
                current_pred = str(prediction_array[0])
                text = "Number: " + str(current_pred)
                position = (500, 80)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(frame, text, position, font, fontScale, color, thickness, cv2.LINE_AA)

            if current_pred == previous_prediction:
                consecutive_frames += 1
            else:
                consecutive_frames = 0
                previous_prediction = current_pred

            if consecutive_frames == frame_threshold:
                current_word += current_pred
                consecutive_frames = 0

        final_text = "Word: " + current_word
        position = (30, 450)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (0,0, 0)
        thickness = 2
        cv2.putText(frame,final_text,position,font, fontScale, color, thickness,cv2.LINE_AA)

        cv2.putText(frame, "Q:Quit", (30,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame, "M:Mode", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "D:Delete", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "B:Back", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Space:Space", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "S:Speak", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Sign Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            if current_mode == "Alphabet":
                current_mode = "number"
            else:
                current_mode = "Alphabet"
        elif key == ord('d'):
            current_word = ""
        elif key == ord("b"):
            current_word = current_word[:-1]
        elif key == ord(' '):
            current_word += " "
        elif key == ord('s'):
            if len(current_word.strip()) > 0:
                threading.Thread(target=speak_word, args=(current_word,)).start()
    cam.release()
    cv2.destroyAllWindows()