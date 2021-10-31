import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

CV_CAP_PROP_POS_FRAMES = 1

def get_estimation_for_frame(video_file, seconds):
    image = get_static_frame(video_file, seconds)
    land_marks = static_images(image)
    frame_vertices = [(land_mark.x, land_mark.y * (-1), land_mark.z) for land_mark in land_marks]
    return frame_vertices

def get_static_frame(video_file, seconds, fps=29.970030):
    cap = cv2.VideoCapture(video_file)
    frame_no = seconds * fps
    cap.set(1, frame_no)
    res, frame = cap.read()
    cap.release()
    return frame

def static_images(image):
    # For static images:
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB))
        # Print handedness and draw hand landmarks on the image.

        # Added content
        frame_land_marks = []
        if results.multi_hand_landmarks:  # returns None if hand is not found
            hand = results.multi_hand_landmarks[
                0]  # results.multi_hand_landmarks returns landMarks for all the hands

            for id, landMark in enumerate(hand.landmark):
                # landMark holds x,y,z ratios of single landmark
                # imgH, imgW, imgC = originalImage.shape  # height, width, channel for image
                # xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
                # landMarkList.append([id, xPos, yPos])
                frame_land_marks.append(landMark)
            # if draw:
            #   mpDraw.draw_landmarks(originalImage, hand, mpHands.HAND_CONNECTIONS)
        return frame_land_marks


def dynamic_images(queue, callback, file=None):
    if file:
        cap = cv2.VideoCapture(file)
    else:
        # For webcam input:
        cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            frame_no = int(cap.get(CV_CAP_PROP_POS_FRAMES))
            success, image = cap.read()
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

            # Added content
            frame_land_marks = []
            if results.multi_hand_landmarks:  # returns None if hand is not found
                hand = results.multi_hand_landmarks[
                    0]  # results.multi_hand_landmarks returns landMarks for all the hands

                for id, landMark in enumerate(hand.landmark):
                    # landMark holds x,y,z ratios of single landmark
                    # imgH, imgW, imgC = originalImage.shape  # height, width, channel for image
                    # xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
                    # landMarkList.append([id, xPos, yPos])
                    frame_land_marks.append(landMark)
                # if draw:
                #   mpDraw.draw_landmarks(originalImage, hand, mpHands.HAND_CONNECTIONS)
                callback(queue, frame_land_marks, frame_no)
            #

            # Draw the hand annotations on the image.
            # image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # if results.multi_hand_landmarks:
            #   for hand_landmarks in results.multi_hand_landmarks:
            #     mp_drawing.draw_landmarks(
            #         image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
