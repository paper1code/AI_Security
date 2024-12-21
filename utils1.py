import copy
import torch.nn.functional as F
import torch
import numpy as np
import os.path as osp
from mmengine.config import Config
from mmengine.runner import Runner
from Grad_cam_utils import GradCAM


def threat_model_predict(model, data_batch, return_logits=False):
    with torch.no_grad():
        predict_data_batch = copy.deepcopy(data_batch)
        predictions = model.test_step(predict_data_batch)
        # data = model.data_preprocessor(predict_data_batch, training=False)
        # predictions = model(**data, mode='predict')

        pre_label = predictions[0].pred_labels.item.item()
        logits = predictions[0].pred_scores.item
        score = (F.softmax(logits.view(1, -1), dim=1)).sort()[0][:, -2:].cpu()
        second_score, pred_score = score[:, 0], score[:, 1]
        del predict_data_batch
    if not return_logits:
        return pre_label, second_score, pred_score
    else:
        return pre_label, second_score, pred_score, logits


def loss_func(logits, labels, perturbations, mask, vid, adv_vid, target_label, untarget=True, lamta1=0.3, lamta2=0.2):
    if untarget:
        sign = -1
        labels = labels
    else:
        sign = 1
        labels[0] = target_label
    CELOSS = F.cross_entropy(logits.view(1, -1), labels)
    loss = sign * CELOSS
    return loss


def get_logits(model, data_batch):
    with torch.no_grad():
        predictions = model.test_step(data_batch)
        logits = predictions[0].pred_scores.item
    return logits


def loss_grad(model, data_batch, mask, vid, lamta1, lamta2, label=None, untarget=True):
    model.zero_grad()
    predict_data_batch = copy.deepcopy(data_batch)
    predict_data_batch['inputs'][0].requires_grad_()
    predictions = model.test_step(predict_data_batch)
    # data = model.data_preprocessor(predict_data_batch, training=False)
    # predictions = model(**data, mode='predict')
    gt_label = predict_data_batch['data_samples'][0].gt_labels.item.cuda()
    logits = predictions[0].pred_scores.item
    perturbations = predict_data_batch['inputs'][0].detach() - vid
    # perturbations = predict_data_batch['inputs'][0]-vid.detach()
    loss = loss_func(logits, gt_label, perturbations, mask, vid, predict_data_batch['inputs'][0], label, untarget,
                     lamta1=lamta1, lamta2=lamta2)

    loss.backward()
    g = predict_data_batch['inputs'][0].grad.detach()
    del predict_data_batch
    torch.cuda.empty_cache()
    return g


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    # -------------------- visualization --------------------
    if args.show or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'VisualizationHook is not set in the `default_hooks` field of ' \
            'config. Please set `visualization=dict(type="VisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    # -------------------- Dump predictions --------------------
    if args.dump is not None:
        assert args.dump.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        dump_metric = dict(type='DumpResults', out_file_path=args.dump)
        if isinstance(cfg.test_evaluator, (list, tuple)):
            cfg.test_evaluator = list(cfg.test_evaluator)
            cfg.test_evaluator.append(dump_metric)
        else:
            cfg.test_evaluator = [cfg.test_evaluator, dump_metric]

    return cfg



def prepare_threat_model(args):
    # load config
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint


    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    model = runner.model
    test_dataloader = runner.test_dataloader
    return model, test_dataloader




def cam_mask(threat_model, data_batch, layer='backbone/layer4/1/relu'):
    data_batch_cam = copy.deepcopy(data_batch)
    # cam = LayerCAM(threat_model, layer)  # I3D
    cam = GradCAM(threat_model, layer)  # I3D
    CAM_imgs, preds = cam.calculate_localization_map(data_batch_cam, use_labels=False)
    del data_batch_cam
    del cam
    CAM_imgs1 = CAM_imgs[0]
    del CAM_imgs

    return CAM_imgs1


def normalize_with_min(data):

    data_copy = data.clone()

    data_copy[data_copy == 0] = float('inf')

    min_val = torch.min(data_copy)

    scale_factor = 1 / min_val

    normalized_data = data * scale_factor
    del data_copy
    return normalized_data

def sparse_input_new1(CAM_imgs, region_rate, len_frame):

    region_mean_list = []
    cam_list = []
    mask_list = []
    total_elements = CAM_imgs.numel()

    elements_to_keep = int(total_elements * (1 - region_rate))

    sorted_tensor, indices = torch.sort(CAM_imgs.view(-1))

    keep_value = sorted_tensor[elements_to_keep]
    for i in range(len(CAM_imgs)):
        cam_img = CAM_imgs[i]
        img = cam_img
        img[img < keep_value] = 0
        list_img = img.repeat(3, 1, 1)

        cam_list.append(list_img)

        total_sum = torch.sum(img)

        nonzero_indices = torch.nonzero(img)
        nonzero_count = nonzero_indices.size(0)

        ele_num = img.numel()
        rate = nonzero_count / ele_num

        region_mean_list.append(rate)

    region_array = np.array(region_mean_list)

    sorted_indices = np.argsort(region_array)[::-1]

    top_values = region_array[sorted_indices[:len_frame]]
    top_indices = sorted_indices[:len_frame]
    print(top_indices)
    zero_frame = torch.zeros_like(cam_list[0])
    for i in range(len(CAM_imgs)):
        if i in top_indices:

            mask_list.append(normalize_with_min(cam_list[i]))
        else:
            mask_list.append(zero_frame)
    mask_list = torch.stack(mask_list)
    del region_mean_list
    del cam_list
    return mask_list.detach()

def get_mask_new1(threat_model, data_batch, layer, region_rate, len_frame):
    cam_imgs = cam_mask(threat_model, data_batch, layer=layer)
    # cam_imgs = cam_imgs.permute(0, 3, 1, 2)
    mask = sparse_input_new1(cam_imgs, region_rate, len_frame)
    return mask

def sign_grad(gs, sign):
    if sign:
        g = torch.sign(gs)
    else:
        g = torch.tensor(gs)
    return g

def Cal_sparse(mask):
    nonzero_indices = torch.nonzero(mask)
    nonzero_count = nonzero_indices.size(0)
    total_num = mask.numel()
    return nonzero_count / total_num

def pertubation(clean, adv):
    loss = torch.nn.L1Loss()
    average_pertubation = loss(clean, adv)
    return average_pertubation


