from src.analytics.analyzer import plot_training_data_histogram
from src.classification.classify_entry import validate, find_hyper_parameters

if __name__ == '__main__':
    # process_video(classify=True)
    # validate()
    # plot_training_data_histogram()
    find_hyper_parameters('RF')