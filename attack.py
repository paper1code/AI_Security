import argparse
import os
import time

import numpy as np
import torch
from mmengine.config import DictAction
from mmengine import track_iter_progress

import copy
from utils import get_mask_new1, loss_grad, prepare_threat_model, \
    threat_model_predict, \
    sign_grad, pertubation, Cal_sparse

torch.cuda.set_device(3)

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args():
    # ------------------------------mmaction2 parameters-------------------------------------#
    parser = argparse.ArgumentParser(
        description='attack')
    parser.add_argument('--config',
                        default='configs/recognition/c3d/c3d_ucf101.py',
                        help='test config file path')
    parser.add_argument('--checkpoint',
                        default='threat_models/c3d_ucf101.pth',
                        help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        default='results',)
    parser.add_argument(
        '--dump',
        type=str)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction)
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=2,
        help='display time of every window. (second)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int,  default=0)
    # ------------------------------mmaction2 parameters-------------------------------------#

    # ------------------------------attack parameters----------------------    ------------------#
    parser.add_argument('--len_frame', type=int, default=4)
    parser.add_argument('--region_rate', type=float, default=0.2)
    parser.add_argument('--len_frame_upper', type=int, default=32)
    parser.add_argument('--region_rate_upper', type=float, default=0.5)
    parser.add_argument('--sparse', type=bool, default=True)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--k', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lamta1',  type=float, default=0)
    parser.add_argument('--lamta2', type=float, default=0)
    parser.add_argument('--epsilon', type=float, default=0.1)

    # ------------------------------attack parameters----------------------------------------#

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args
def untargeted_white_box_attack(data_batch, threat_model, layer, region_rate, len_frame, k, lr, random_start,
                                lamta1,lamta2,epsilon=0.05):
    global i
    adv_data_batch = copy.deepcopy(data_batch)
    sign = True
    vid = adv_data_batch['inputs'][0].cuda()
    gt_label = adv_data_batch['data_samples'][0].gt_labels.item.item()
    if random_start:
        mask = get_mask_new1(threat_model, adv_data_batch, layer, region_rate, len_frame).unsqueeze(0)
        mask = mask.permute(0,2,1,3,4).cuda()
        perturb = torch.randn_like(vid) * (2 * epsilon) - epsilon
        mask = mask.to(torch.float32)
        adv_vid = vid.detach() + perturb*mask.detach()
        adv_vid = torch.clamp(adv_vid, 0., 1.)
        adv_data_batch['inputs'][0] = adv_vid
    else:
        adv_vid = vid
    for i in range(k):
        torch.cuda.empty_cache()
        g = loss_grad(threat_model, adv_data_batch, mask.detach(), vid.detach(), lamta1, lamta2)
        adv_vid = adv_vid - lr * sign_grad(g, sign).detach() * mask
        adv_vid = torch.clamp(adv_vid, vid - epsilon, vid + epsilon)
        adv_vid = torch.clamp(adv_vid, 0., 1.)
        adv_data_batch['inputs'][0] = adv_vid
    pre_label, second_score, pred_score = threat_model_predict(threat_model, adv_data_batch)
    if pre_label == gt_label:
        res = False
    else:
        res = True
    ap = pertubation(vid, adv_vid)
    sparse_rate = Cal_sparse(mask)
    del adv_data_batch
    torch.cuda.empty_cache()
    return res, adv_vid, ap, 0,sparse_rate,i

def targeted_white_box_attack(data_batch, threat_model, layer, region_rate, len_frame, k, lr, random_start,
                                lamta1,lamta2,target_label,epsilon=0.05):
    global i
    adv_data_batch = copy.deepcopy(data_batch)
    sign = True
    vid = adv_data_batch['inputs'][0].cuda()
    if random_start:
        mask = get_mask_new1(threat_model, adv_data_batch, layer, region_rate, len_frame).unsqueeze(0)
        mask = mask.permute(0,2,1,3,4).cuda()
        perturb = torch.randn_like(vid) * (2 * epsilon) - epsilon
        mask = mask.to(torch.float32)
        adv_vid = vid.detach() + perturb*mask.detach()
        adv_vid = torch.clamp(adv_vid, 0., 1.)
        adv_data_batch['inputs'][0] = adv_vid
    else:
        adv_vid = vid
    for i in range(k):
        torch.cuda.empty_cache()
        g = loss_grad(threat_model, adv_data_batch, mask.detach(), vid.detach(), lamta1, lamta2,label=target_label,untarget=False)
        adv_vid = adv_vid - lr * sign_grad(g, sign).detach() * mask
        adv_vid = torch.clamp(adv_vid, vid - epsilon, vid + epsilon)
        adv_vid = torch.clamp(adv_vid, 0., 1.)
        adv_data_batch['inputs'][0] = adv_vid
    pre_label, second_score, pred_score = threat_model_predict(threat_model, adv_data_batch)
    if pre_label != target_label:
        res = False
    else:
        res = True
    ap = pertubation(vid, adv_vid)
    sparse_rate = Cal_sparse(mask)
    del adv_data_batch
    torch.cuda.empty_cache()
    return res, adv_vid, ap, 0,sparse_rate,i


def main():
    args = parse_args()
    threat_model, test_dataloader = prepare_threat_model(args)
    threat_model.eval()

    spatial_threshold_upper = args.region_rate_upper
    temporal_threshold_upper = args.len_frame_upper


    success_num = 0
    sum = 0
    total_ap = 0
    total_iter = 0
    total_time = 0
    total_ssim = 0
    total_sparse = 0
    total_times = 0
    data_flag_set = set()
    log_path = os.path.join(args.work_dir,'c3d_ucf101_log_lr_{}_epsilon_{}_len_{}_region_{}_new1_untarget.txt'.format(args.lr,args.epsilon,args.len_frame,args.region_rate))
    log_file = open(log_path,'w')
    layer_name = 'backbone/conv5b/activate'
    for data_batch in track_iter_progress(test_dataloader):
        res = False
        times = 0
        spatial_threshold = args.region_rate
        temporal_threshold = args.len_frame
        torch.cuda.empty_cache()
        data_batch_input = copy.deepcopy(data_batch)
        ori_input_tensor = data_batch_input['inputs'][0]
        ori_input_tensor = ori_input_tensor.type(torch.float)
        ori_input_tensor /= 255  # normalization
        ori_input_tensor = ori_input_tensor.cuda()
        data_batch_input['inputs'][0] = ori_input_tensor
        gt_label = data_batch['data_samples'][0].gt_labels.item.item()
        target_label = (gt_label+10) % 51
        if gt_label in data_flag_set:
            print("have tested this class:{}".format(gt_label))
            del data_batch_input
            continue
        pre_label, second_score, pred_score = threat_model_predict(threat_model, data_batch_input)  # just correct example
        if pre_label != gt_label:
            print("wrong example,next example")
            del data_batch_input
            continue

        start_time = time.time()
        while times<5 and res is False:  # Dynamic
            res, adv_vid, ap,ssim,sparse,iter =untargeted_white_box_attack(data_batch_input, threat_model,layer_name,spatial_threshold,temporal_threshold,args.k,args.lr,True,args.lamta1,args.lamta2,args.epsilon)
            # res, adv_vid, ap,ssim,sparse,iter =targeted_white_box_attack(data_batch_input, threat_model,'backbone/layer4/2/relu',spatial_threshold,temporal_threshold,args.k,args.lr,True,args.lamta1,args.lamta2,target_label,args.epsilon)
            if res is False:
                spatial_threshold += np.power(2.0,times-1)*0.05
                spatial_threshold = min(spatial_threshold,spatial_threshold_upper)
                temporal_threshold += np.power(2.0,times)*2
                temporal_threshold = min(temporal_threshold,temporal_threshold_upper)
                temporal_threshold = int(temporal_threshold)
                times += 1
        end_time = time.time()
        execution_time = end_time - start_time
        if res:
            sum += 1
            success_num += 1
            total_iter += iter
            total_ap += ap.item()
            total_time += execution_time
            total_ssim += ssim
            total_sparse += sparse
            total_times += (times+1)
            print('\nexample is {} '.format(sum))
            print('\navg ap is {} '.format(total_ap / success_num))
            print('\navg sparse is {} '.format(total_sparse / success_num))
            print('\nsuccess rate  {} '.format(success_num / sum))
            print('\ntimes is {} '.format(total_times/sum))
            log_file.write('\nexample is {} '.format(sum))
            log_file.write('\navg ap is {} '.format(total_ap / success_num))
            log_file.write('\navg sparse is {} '.format(total_sparse / success_num))
            log_file.write('\nsuccess rate  {} '.format(success_num / sum))
            log_file.write('\ntimes is {} '.format(total_times/sum))
            log_file.flush()
            data_flag_set.add(gt_label)
        else:
            sum += 1
            print("\nexample {} fails, ap is {}".format(sum, ap))
            data_flag_set.add(gt_label)
        if sum == 51:
            break
        del data_batch_input
    log_file.close()


if __name__ == '__main__':
    main()
