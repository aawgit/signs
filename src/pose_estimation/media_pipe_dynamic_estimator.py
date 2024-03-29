import time as tm_mod

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from src.feature_extraction.pre_processor import pre_process_single_frame, flatten_points

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

CV_CAP_PROP_POS_FRAMES = 1


def static_images(file_list):
    # For static images:
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
        for idx, file in enumerate(file_list):
            # Read an image, flip it around y-axis for correct handedness output (see
            # above).
            image = cv2.flip(cv2.imread(file), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print handedness and draw hand landmarks on the image.
            print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
                continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                print('hand_landmarks:', hand_landmarks)
                print(
                    f'Index finger tip coordinates: (',
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                )
                mp_drawing.draw_landmarks(
                    annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.imwrite(
                'annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))

def _temp_dynamic_images(means, file=None):
    if file:
        cap = cv2.VideoCapture(file)
    else:
        # For webcam input:
        cap = cv2.VideoCapture(0)
    frames = 0
    tot_du = 0

    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=1
    ) as hands:

        skip = False
        while cap.isOpened():
            frame_no = 1#int(cap.get(CV_CAP_PROP_POS_FRAMES))
            success, image = cap.read()
            frames = frames + 1
            # if skip:
            #     skip = False
            #     continue
            # else:
            #     skip = True
            start = tm_mod.time()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)
            results = mp_callback_static(results)
            if not results: continue
            land_marks, angles = pre_process_single_frame(results)
            lm_row = []
            for landmark_point in land_marks:
                lm_row.extend(np.round(landmark_point, 4))
            # means_internal = means.drop(['sign', 'source'], axis=1)
            match: pd.DataFrame = means[(means['1_0'] == (lm_row[3])) &
                                        (means['1_1'] == (lm_row[4])) &
                                        (means['1_2'] == (lm_row[5]))
            ]
            if not match.empty:
                x = 5
                print(match['sign'].iloc[0], frames)
            else:
                x= 0

def mp_callback_static(results):
    frame_land_marks = []
    if results.multi_hand_landmarks:  # returns None if hand is not found
        hand = results.multi_hand_landmarks[
            0]  # results.multi_hand_landmarks returns landMarks for all the hands
        left_hand = results.multi_handedness[0].classification[0].label == 'Left'
        for id, landMark in enumerate(hand.landmark):
            # landMark holds x,y,z ratios of single landmark
            # imgH, imgW, imgC = originalImage.shape  # height, width, channel for image
            # xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
            # landMarkList.append([id, xPos, yPos])
            if left_hand:
                land_mark = (landMark.x, landMark.y, landMark.z)
            else:
                land_mark = (landMark.x, landMark.y, (-1)*landMark.z)
            frame_land_marks.append(land_mark)
            # if draw:
    return frame_land_marks



def dynamic_images(queue, callback, file=None):
    if file:
        cap = cv2.VideoCapture(file)
    else:
        # For webcam input:
        cap = cv2.VideoCapture(0)
    frames = 0
    tot_du = 0

    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=1
    ) as hands:

        skip = False
        while cap.isOpened():
            frame_no = 1#int(cap.get(CV_CAP_PROP_POS_FRAMES))
            success, image = cap.read()
            frames = frames + 1
            # if skip:
            #     skip = False
            #     continue
            # else:
            #     skip = True
            start = tm_mod.time()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            end = tm_mod.time()
            duration = end - start
            tot_du = tot_du + duration

            callback(queue, results, frame_no)

            cv2.imshow('Video stream', image)
            if cv2.waitKey(5) & 0xFF == 27:
                print(frames/tot_du)
                break
    cap.release()
    print(frames / tot_du)
