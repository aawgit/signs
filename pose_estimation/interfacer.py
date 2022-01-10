import logging
from pose_estimation.media_pipe_dynamic_estimator import dynamic_images
from pose_estimation.pose_estimator_by_frame import static_images
from pose_estimation.adjuster import adjust_finger_bases
from pose_estimation.media_pipe_static_estimator import static_images_2


def mp_callback(queue, results, frame_no):
    # frame_vertices = [(land_mark.x, land_mark.y * (-1), (-1)*land_mark.z) for land_mark in frame_land_marks]
    # # frame_vertices = adjust_finger_bases(frame_vertices)
    #

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
                land_mark = (landMark.x, landMark.y, (-1) * landMark.z)
            frame_land_marks.append(land_mark)
            # if draw:
        queue.put((frame_land_marks, frame_no))


def mp_estimate_pose(que, file=None):
    # Puts a stream of hand poses in to the queue
    logging.info('Initiating pose estimation...')
    dynamic_images(que, mp_callback, file)


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

def mp_estimate_pose_static(image):
    logging.info('Initiating pose estimation...')
    return static_images(image, mp_callback_static)

def mp_estimate_pose_from_image(path):
    result = static_images_2(path)
    return mp_callback_static(result)