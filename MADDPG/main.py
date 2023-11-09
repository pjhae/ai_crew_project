# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch
# File func: main func
import datetime
from train import train
from utility.arguments import parse_args
from utility.utils import VideoRecorder

if __name__ == '__main__':
    arglist = parse_args()   
    
    # For video
    video_directory = './video/{}'.format(datetime.datetime.now().strftime("%H:%M:%S %p"))
    video = VideoRecorder(dir_name = video_directory)

    train(arglist, video)
