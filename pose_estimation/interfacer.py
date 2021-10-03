from pose_estimation.media_pipe_dynamic_estimator import dynamic_images
from pose_estimation.adjuster import adjust_finger_bases

def mp_callback(queue, frame_land_marks, frame_no):
    frame_vertices = [(land_mark.x, land_mark.y * (-1), land_mark.z) for land_mark in frame_land_marks]
    frame_vertices = adjust_finger_bases(frame_vertices)
    queue.put((frame_vertices, frame_no))


def mp_estimate_pose(que, file=None):
    # Puts a stream of hand poses in to the queue
    dynamic_images(que, mp_callback, file)