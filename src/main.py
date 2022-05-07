from src.classification.dynamic_classifier import dtw_test

from src.classification.classify_entry import process_video

if __name__ == '__main__':
    process_video(classify=True)
    # dtw_test()