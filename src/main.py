from src.classification.classify_entry import process_video

if __name__ == '__main__':
    video = None
    # Uncomment below line and give a vide file location to classify it
    # video = "./data/subject10/fafed337-1a41-4f55-b318-ed3e1b14e059.mp4"
    process_video(video, classify=True)
