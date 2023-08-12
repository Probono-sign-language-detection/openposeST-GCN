import sys
import os
from glob import glob
from tqdm import tqdm

# Set the path for the OpenPose library
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/openpose/build/python/openpose/Release')
os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/openpose/build/x64/Release;' + dir_path + '/openpose/build/bin;'

import pyopenpose as op
import cv2
import argparse

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="sign.jpg")
# parser.add_argument("--video", default="testiloveyou.mp4")
args = parser.parse_known_args()

# Set up OpenPose
params = dict()
params["model_folder"] = r"C:\Users\JeongSeongYun\Desktop\openposetest\pyopen\openpose\models/"
params["hand"] = True
params['write_json'] = r"C:\Users\JeongSeongYun\Desktop\openposetest\pyopen\output_json/"
params['net_resolution'] = '320x176'
params['face_net_resolution'] = '320x320'
TEST_IMG_ROOT = "/test_imgs/*.jpg" # 수정하시면 됩니다. OS로 좀 더 범용적으로 바꾸는 게 좋긴 할듯
img_lst = glob(TEST_IMG_ROOT)

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()



"""
folder 내의 모든 이미지에 대해 Skeleton 추론을 실시한 뒤, 이에 대한 정보를 저장합니다
"""

TEST_IMG_ROOT = r"./test_imgs/*.jpg" # 수정하시면 됩니다. OS로 좀 더 범용적으로 바꾸는 게 좋긴 할듯
img_lst = glob(TEST_IMG_ROOT)
print(f"✅\t# of Frames: {len(img_lst)}")

def get_skeleton_inputs(TEST_IMG_ROOT):
    print("Extracting Skeleton Datas from Frames...")
    for img in tqdm(img_lst):
        datum = op.Datum()
        img2process = cv2.imread(img)
        datum.cvInputData = img2process
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    print("🚀 Getting Skeleton Datas is Done! 🚀")
    return params['write_json']

if __name__ == "__main__":

    for img in tqdm(img_lst):
        datum = op.Datum()
        img2process = cv2.imread(img)
        datum.cvInputData = img2process
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))



