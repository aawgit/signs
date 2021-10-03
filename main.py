import logging
import multiprocessing

from feature_extraction.pre_processor import run_pre_process_steps
from feature_extraction.renderer import render
from pose_estimation.interfacer import mp_estimate_pose

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=3)
    m = multiprocessing.Manager()
    hand_pose_q = m.Queue()

    file = "./data/video/SLSL - Sinhala Sign Alphabet - Sri Lankan Sign Language - Chaminda Hewapathirana.mp4"
    # file = None
    # file = "./data/video/yt1s.com - SLSL1Sinhala Manual Alphabet_360p.mp4"
    # file = "/home/aka/Downloads/ego hands dataset/videos_1/Subject04/Scene2/Color/rgb2.avi"

    workers_1 = pool.apply_async(mp_estimate_pose, (hand_pose_q, file))

    pre_processed_q_1 = m.Queue()
    pre_processed_q_2 = m.Queue()
    workers_2 = pool.apply_async(run_pre_process_steps, (hand_pose_q, pre_processed_q_1, pre_processed_q_2))
    # workers_3 = pool.apply_async(labeller_worker, (pre_processed_q_2,))
    # workers_3 = pool.apply_async(classifier_by_vertices_worker, (pre_processed_q_2,))

    render(pre_processed_q_1)
