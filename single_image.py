import cv2
import torch

from lib.core.config import config, update_config
from lib.dataset.adafuse_collate import adafuse_collate
from lib.dataset.unrealcv_dataset import UnrealcvData
from lib.models.adafuse_network import get_multiview_pose_net
from lib.models.pose_resnet import get_pose_net
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from lib.utils.transforms import get_affine_transform

update_config("./experiments/occlusion_person/occlusion_person_8view_cpu.yaml")

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

backbone_model = get_pose_net(config, is_train=False)
model = get_multiview_pose_net(
    backbone_model, config)

model_file_path = config.NETWORK.ADAFUSE
model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')), strict=True)
model.eval()




def process_video(video_filepath):
    cap = cv2.VideoCapture(video_filepath)
    grabbed, frame = cap.read()

    while grabbed:
        grabbed, frame = cap.read()


        trans = get_affine_transform(center, scale, rotation, frame.image_size)
        # ! Notice: this trans represents full image to cropped image,
        # not full image->heatmap
        input = cv2.warpAffine(
            frame,
            trans, (int(frame.image_size[0]), int(frame.image_size[1])),
            flags=cv2.INTER_LINEAR)

        digestible = transform(frame)
        res = model(digestible, run_phase="test")

    cap.release()


process_video("./PXL_20230719_1603230853.mp4")