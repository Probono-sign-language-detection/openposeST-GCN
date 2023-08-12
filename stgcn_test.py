import os
from glob import glob

from configs import Config
from tgcn_model import GCN_muti_att

import numpy as np
import torch

from return_skeletons import get_skeleton_inputs
from dataset import ProbonoSignDataset

print("✅\tImport is Done!")

def get_models():
    root = r"C:\Users\JeongSeongYun\Desktop\openposetest\WLASL\code\TGCN/"
    trained_on = 'asl2000'
    pretrained_model = "ckpt.pth"

    num_frames = len(glob(r"C:\Users\JeongSeongYun\Desktop\openposetest\pyopen\output_json/*"))

    ## 이미 추출했으면 그냥 건너기
    if num_frames == 0:
        frame_file_root = get_skeleton_inputs(TEST_IMG_ROOT=r"C:\Users\JeongSeongYun\Desktop\openposetest\pyopen\test_imgs\*.jpg")
    else:
        frame_file_root = r"C:\Users\JeongSeongYun\Desktop\openposetest\pyopen\output_json"

    config_file = os.path.join(root, f"configs/{trained_on}/{pretrained_model}.ini")
    configs = Config(r"C:\Users\JeongSeongYun\Desktop\openposetest\WLASL\code\TGCN\archived\asl2000\asl2000.ini")
    num_samples = configs.num_samples
    hidden_size = configs.hidden_size
    drop_p = configs.drop_p
    num_stages = configs.num_stages
    batch_size = configs.batch_size

    dataset = ProbonoSignDataset(pose_folder=frame_file_root)

    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=16,
                                            )

    ## Config Parameter는 다시 확인할 필요가 있음!
    model = GCN_muti_att(input_feature=num_samples * 2, hidden_feature=hidden_size,
                        num_class=int(trained_on[3:]), p_dropout=drop_p, num_stage=num_stages)
    
    ckp = torch.load(os.path.join(root, f"archived/{trained_on}/{pretrained_model}"),
                    map_location=torch.device("cpu"))
    model.load_state_dict(ckp)

    return model, data_loader


def inference(model, test_loader):
    print("(❁´◡`❁) Start Inference... (❁´◡`❁)")
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            xs = data
            # CPU환경을 가정하므로 cuda에 올리는 건 제외했습니다.. 나중에 여차하면 device설정으로 올려버리기!
            output = model(xs)
            # print("model(xs) : ",type(output), output.size()) # 4, 2000 ... 흠... 다시 생각해보기
            # print(output)
            output = torch.mean(output, dim=0)
            # print("mean(output) : ", type(output), output.size())
            # print(output)
            y_pred = output.max(0, keepdim=True)

    print(f"🤖 My Prediction: {y_pred[1]} 🤖")


if __name__ == "__main__":
    model, loader = get_models()

    inference(model, loader)
