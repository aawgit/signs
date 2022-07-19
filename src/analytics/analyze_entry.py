from matplotlib import pyplot as plt

from src.feature_extraction.pre_processor import pre_process_single_frame
from src.pose_estimation.interfacer import mp_estimate_pose_static
from src.pose_estimation.media_pipe_dynamic_estimator import static_images
from src.classification.classify_entry import get_training_data
from src.pose_estimation.vertices_mapper import EDGES_MEDIA_PIPE
from src.utils.video_utils import video_meta, get_static_frame2


def get_saved_land_mark(sign, sign_file_df, candidate_no=0, source=None):
    if not source:
        filtered_df = sign_file_df[sign_file_df['sign'] == sign].drop('source', axis=1, errors='ignore')
    else:
        filtered_df = sign_file_df[(sign_file_df['sign'] == sign) & (sign_file_df['source'] == source)]. \
            drop('source',
                 axis=1,
                 errors='ignore')
    coordinates = list(filtered_df.iloc[candidate_no])[1:]
    land_mark = [tuple(coordinates[i:i + 3]) for i in range(0, len(coordinates), 3)]
    return land_mark

def save_landmark_plot(label: str, landmark: list, plot_folder: str, file_name: str):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection='3d'))
    # views = [(-90, 90), (0, 90), (0, 0), (45, 90)]

    axs.set_xlim3d(-1, 1)
    axs.set_ylim3d(0, 2)
    axs.set_zlim3d(-1, 1)
    axs.view_init(-15, 0)
    axs.set_xlabel('$X$', fontsize=20)
    axs.set_ylabel('$Y$')
    axs.set_zlabel('$Z$')
    un_flattened = landmark
    zdata = [point[2] for point in un_flattened]
    xdata = [point[0] for point in un_flattened]
    ydata = [point[1] for point in un_flattened]
    axs.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    plt.axis('off')
    for point_pair in EDGES_MEDIA_PIPE:
        first = point_pair[0]
        second = point_pair[1]
        axs.plot3D([xdata[first], xdata[second]], [ydata[first], ydata[second]], [zdata[first], zdata[second]], 'b')
    plt.savefig("{}/{}-{}.png".format(plot_folder, label, file_name))

if __name__ == '__main__':
    # means = get_training_data()

    video_m = video_meta.get(1)
    video = video_m.get('location')
    fps = video_m.get('fps')
    frame = 2187
    #
    # time = 150.658561296859  # testing th, estimations seem flat
    ref_sign = 27
    # saved_lm = get_saved_land_mark(ref_sign, means, 2)
    # saved_lm, angles = pre_process_single_frame(saved_lm)
    # render_static(saved_lm)

    # saved_lm2 = get_saved_land_mark(ref_sign, means, 4)

    image = get_static_frame2(video, frame)
    # image = cv2.imread('/home/aka/Downloads/20220128_145109.jpg')
    # land_marks = mp_estimate_pose_from_image('/home/aka/Downloads/14-03_3/20220314_231741.jpg')
    # land_marks, angles = pre_process_single_frame(land_marks)
    # # #
    # flatted = flatten_points(land_marks)
    # rounded = [np.round(p, 4) for p in flatted]
    # print(rounded)
    #
    # video_m = video_meta.get(2)
    # video = video_m.get('location')
    # fps = video_m.get('fps')

    # time = 1 + 48
    # image2 = get_static_frame2(video, frame)
    land_marks2 = mp_estimate_pose_static(image)
    land_marks2, angles2 = pre_process_single_frame(land_marks2)
    # # #
    # render_static(land_marks)

    # render_static_2_hands(saved_lm, land_marks2)
    save_landmark_plot('', land_marks2, '', '')
    # static_images(['/home/aka/Downloads/20220329_100524.jpg'])