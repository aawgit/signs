import logging

import cv2


def crop_video_and_save(file, frame_rate=25.0):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(file.split('.')[0] + '_cropped.mp4', fourcc, 20.00, (640, 720))

    cap = cv2.VideoCapture(file)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image = image[:, 640: 1280, :]
        out.write(image)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break


def show_frame(frame):
    cv2.imshow('MediaPipe Hands', frame)
    cv2.waitKey()


def get_static_frame(video_file, seconds, fps=29.970030):
    cap = cv2.VideoCapture(video_file)
    frame_no = seconds * fps
    cap.set(1, frame_no)
    res, frame = cap.read()
    cap.release()
    return frame


video_meta = {
    1: dict(location="./data/video/SLSL - Sinhala Sign Alphabet - Sri Lankan Sign Language - Chaminda Hewapathirana.mp4",
            fps=29.97),
    2: dict(location="./data/video/yt1s.com - SLSL1Sinhala Manual Alphabet_360p.mp4",
            fps=29.97),
    3: dict(location="./data/video/buddika/a-uu.mp4",
            fps=29.59),
    4: dict(location="./data/video/Geshani.mp4",
            fps=20),
    5: dict(location= "/home/aka/Downloads/ego hands dataset/videos_1/Subject04/Scene2/Color/rgb2.avi",
            fps=30)
}

if __name__ == '__main__':
    logging.info('Starting video playback...')
    video = "../data/video/SLSL - Sinhala Sign Alphabet - Sri Lankan Sign Language - Chaminda Hewapathirana.mp4"
    cap = cv2.VideoCapture(video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    def onChange(trackbarValue):
        cap.set(cv2.CAP_PROP_POS_FRAMES, trackbarValue)
        err, img = cap.read()
        cv2.imshow("mywindow", img)
        pass


    cv2.namedWindow('mywindow')
    cv2.createTrackbar('start', 'mywindow', 0, length, onChange)
    cv2.createTrackbar('end', 'mywindow', 100, length, onChange)

    onChange(0)
    cv2.waitKey()

    start = cv2.getTrackbarPos('start', 'mywindow')
    end = cv2.getTrackbarPos('end', 'mywindow')
    if start >= end:
        raise Exception("start must be less than end")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    while cap.isOpened():
        err, img = cap.read()
        if cap.get(cv2.CAP_PROP_POS_FRAMES) >= end:
            break
        cv2.imshow("mywindow", img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
