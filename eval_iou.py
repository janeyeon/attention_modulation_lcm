import numpy as np
import os
import yaml
import pickle
import argparse
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from utils.general import non_max_suppression_mask_conf

# To install detectron2, refer to https://detectron2.readthedocs.io/en/latest/tutorials/install.html
# python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='LCM', choices=['LCM', 'SD', 'Turbo'])
    parser.add_argument('-w', '--woModulation', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    
    model_name = args.model_name
    cond = 'creg1_test' if args.woModulation else 'woModulation'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open('./hyp.scratch.mask.yaml') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    weigths = torch.load('yolov7-mask.pt')
    model = weigths['model']
    model = model.half().eval().to(device)

    with open('./dataset/testset_instances.pkl', 'rb') as f:
        inst_gt = pickle.load(f) 

    trans = transforms.Compose([transforms.Resize(224), 
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

    iou = []
    for i in tqdm(range(len(inst_gt))):
        for repeat in ('0_0', '0_1', '1_0', '1_1'):
            # preprocess
            im_path = glob(f"./outputs/{i:02}/{model_name}*idx{i:02}*{cond}*{repeat}*")[0]
            cls_gt = inst_gt[i]['cls_gt']
            mask_gt = inst_gt[i]['mask_gt']

            image = trans(Image.open(im_path)).unsqueeze(0)
            image = image.half().to(device)

            # predict instance masks and classes
            output = model(image)
            inf_out, attn, bases, sem_output = output['test'], output['attn'], output['bases'], output['sem']
            bases = torch.cat([bases, sem_output], dim=1)
            nb, _, height, width = image.shape
            names = model.names
            pooler_scale = model.pooler_scale
            pooler = ROIPooler(output_size=hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1,\
                               pooler_type='ROIAlignV2', canonical_level=2)

            output, output_mask, output_mask_score, _, _ = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp,
                                                                                         conf_thres=0.5, iou_thres=0.65,
                                                                                         merge=False, mask_iou=None)
            pred, mask_pred = output[0], output_mask[0]
            base = bases[0]
            if pred == None:
                iou.append(0)
                continue

            bboxes = Boxes(pred[:, :4])
            original_mask_pred = mask_pred.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
            mask_pred = retry_if_cuda_oom(paste_masks_in_image)(original_mask_pred, bboxes, (height, width), threshold=0.5)
            mask_pred = F.interpolate(mask_pred.float().unsqueeze(1),(64,64),
                                      mode='bicubic',align_corners=False).squeeze(1).detach().cpu().numpy()
            cls_pred = pred[:, 5].detach().cpu().numpy()
            cls_txt_pred = [names[int(p)] for p in cls_pred]
            pred_conf = pred[:, 4].detach().cpu().numpy()

            # calculate iou (recall)
            cur_iou = []
            for p in range(len(cls_gt)):
                if cls_gt[p] in cls_txt_pred:
                    curidx = cls_txt_pred.index(cls_gt[p])
                    intersection = np.logical_and(mask_gt[p], mask_pred[curidx])
                    union = np.logical_or(mask_gt[p], mask_pred[curidx])
                    cur_iou.append(np.sum(intersection) / np.sum(union))
                    del cls_txt_pred[curidx]
                    mask_pred = np.concatenate([mask_pred[:curidx,:,:], mask_pred[curidx+1:,:,:]], 0)
                else:
                    cur_iou.append(0)
            iou.append(np.mean(cur_iou))

    mean_iou = np.mean(iou)        
    print(mean_iou)
    with open(f"./outputs/iou_{model_name}_test{'_woModulation'*args.woModulation}.txt", 'w') as f:
        f.write(str(mean_iou))