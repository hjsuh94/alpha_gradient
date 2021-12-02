import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import argparse
import shutil
import subprocess
from tqdm import tqdm

import os

def draw_frame(state, filename):
    plt.figure(figsize=(8,8))

    length = 9.0
    radius = 1.0
    theta = state[0]

    xpos = length * np.sin(theta)
    ypos = -length * np.cos(theta)

    plt.plot([0, xpos], [0, ypos], 'k-', linewidth=2.0)

    circle = plt.Circle((xpos, ypos), radius, color='springgreen')
    plt.gca().add_patch(circle)    

    circle = plt.Circle((0, 0), 0.3, color='black')
    plt.gca().add_patch(circle)    

    plt.axis('equal')
    plt.xlim([-12, 12])
    plt.ylim([-12, 12])


    plt.savefig(filename)
    plt.close()

def dump_images(x_trj, temp_dir):
    T = x_trj.shape[0]
    for t in tqdm(range(T)):
        filename = "image_{:05d}".format(t)
        draw_frame(x_trj[t,:], os.path.join(temp_dir, filename))

def save_video(temp_dir, file_dir, video_name, framerate=30):
    image_name = os.path.join(temp_dir, "image_%5d.png")
    video_name = os.path.join(file_dir, video_name)
    subprocess.call([
        'ffmpeg', '-framerate', str(framerate), '-i',
        image_name, '-r', str(framerate), '-pix_fmt', 'yuv420p',
        video_name
    ])    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help=".npy trajectory file to animate.")
    args = parser.parse_args()
    file_path = args.filename

    x_trj = np.load(file_path)
    file_dir = os.path.split(file_path)[0]
    file_name = os.path.split(file_path)[1]
    temp_dir = os.path.join(file_dir, "temp")

    videofile_name = file_name[:-4] + ".mp4"

    os.mkdir(temp_dir)

    dump_images(x_trj, temp_dir)
    save_video(temp_dir, file_dir, videofile_name)

    shutil.rmtree(temp_dir)



