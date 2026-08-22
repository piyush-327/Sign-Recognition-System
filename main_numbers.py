import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import csv

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

    csv_filename = "dataset_numbers.csv"
    header = ["Label"]
    for i in range(21):
        header.append(f"X{i}")
        header.append(f"Y{i}")
        header.append(f"Z{i}")

    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    is_recording = False
    saved_frame_count = 0
    numbers = ['1', '2', '3', '4', '5', '6','7','8','9','0']
    current_number_index = 0
    target_label = numbers[current_number_index]

    while True:
        success, frame = cam.read()
        if not success:
            print("Failed to Load Camera")
            print("Exiting Program")
            exit()

        frame = cv2.flip(frame,1)

        # fps = cam.get(cv2.CAP_PROP_FPS)
        # frame_id = cam.get(cv2.CAP_PROP_POS_FRAMES)
        # calculated_time_sec = frame_id / fps if fps > 0 else 0
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

        if is_recording == False:

            text = "Current Number: " + str(target_label)
            position = (30, 80)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 255, 255)
            thickness = 2
            cv2.putText(frame, text, position, font, fontScale, color, thickness, cv2.LINE_AA)

            text = "R to Record"
            position = (30, 120)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 255, 255)
            thickness = 2
            cv2.putText(frame, text, position, font, fontScale, color, thickness, cv2.LINE_AA)

            text = "N for Next"
            position = (30, 150)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 255, 255)
            thickness = 2
            cv2.putText(frame, text, position, font, fontScale, color, thickness, cv2.LINE_AA)

        if is_recording == True:
            text = "Current Number: " + str(target_label)
            position = (30, 80)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 255, 255)
            thickness = 2
            cv2.putText(frame, text, position, font, fontScale, color, thickness, cv2.LINE_AA)

            text = "Recording..."
            position = (30, 120)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 255, 255)
            thickness = 2
            cv2.putText(frame, text, position, font, fontScale, color, thickness, cv2.LINE_AA)

            text = "N for Next"
            position = (30, 150)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 255, 255)
            thickness = 2
            cv2.putText(frame, text, position, font, fontScale, color, thickness, cv2.LINE_AA)

            dataset = [target_label] + normalized_coordinates
            with open("dataset_numbers.csv", mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(dataset)
                saved_frame_count += 1

            if saved_frame_count >= 200:
                is_recording = False
                print(f"Finished recording 200 frames for {target_label}")

        cv2.imshow("Sign Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            is_recording = True
            saved_frame_count = 0
        elif key == ord('n') and current_number_index < len(numbers) - 1:
            if current_number_index < len(numbers):
                current_number_index += 1
                target_label = numbers[current_number_index]
                is_recording = False
            else:
                print("End of Alphabets Reached")
    cam.release()
    cv2.destroyAllWindows()