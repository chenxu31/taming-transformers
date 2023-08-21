# -*- coding: utf-8 -*-

import argparse
import importlib
import os
import sys
import pdb
import torch
import time
import numpy
import platform
import skimage.io
import glob
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel


if platform.system() == 'Windows':
    UTIL_DIR = r"E:\我的坚果云\sourcecode\python\util"
else:
    UTIL_DIR = r"/home/chenxu/我的坚果云/sourcecode/python/util"

sys.path.append(UTIL_DIR)
import common_metrics
import common_cmf_pt as common_cmf
import common_net_pt as common_net


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def main(device, args):
    ckpt_file = os.path.join(args.log_dir, "checkpoints", "last.ckpt")
    config_files = sorted(glob.glob(os.path.join(args.log_dir, "configs", "*.yaml")))
    configs = [OmegaConf.load(cfg) for cfg in config_files]
    config = OmegaConf.merge(*configs).model
    model = VQModel(ddconfig=config.params.ddconfig,
                    lossconfig=config.params.lossconfig,
                    n_embed=config.params.n_embed,
                    embed_dim=config.params.embed_dim,
                    ckpt_path=ckpt_file)
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model.to(device)

    if args.modality == "mri":
        test_data, _, _ = common_cmf.load_test_data(args.data_dir)
    elif args.modality == "ct":
        _, test_data, _ = common_cmf.load_test_data(args.data_dir)
    else:
        assert 0
    patch_shape = (config.params.ddconfig.in_channels, test_data.shape[2], test_data.shape[3])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    psnr_list = numpy.zeros((test_data.shape[0],), numpy.float32)
    with torch.no_grad():
        for i in range(test_data.shape[0]):
            syn_im = common_net.produce_results(device, model, [patch_shape, ], [test_data[i], ],
                                                data_shape=test_data.shape[1:], patch_shape=patch_shape, is_seg=False,
                                                batch_size=16)
            syn_im = syn_im.clip(-1, 1)
            psnr_list[i] = common_metrics.psnr(syn_im, test_data[i])

            if args.output_dir:
                common_cmf.save_nii(syn_im, os.path.join(args.output_dir, "syn_%d.nii.gz" % i))

    msg = "psnr:%f/%f" % (psnr_list.mean(), psnr_list.std())
    print(msg)

    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write(msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=int, default=0, help="gpu device id")
    parser.add_argument('--data_dir', type=str, default=r'/home/chenxu/datasets/pelvic/h5_data_nonrigid/', help='path of the dataset')
    parser.add_argument('--log_dir', type=str, default=r'/home/chenxu/training/logs/taming/ae_ct_vq/2023-01-09T21-58-22_pelvic_vqgan', help="checkpoint file dir")
    parser.add_argument('--output_dir', type=str, default='/home/chenxu/training/test_output/taming/ae_ct_vq', help="the output directory")
    parser.add_argument('--modality', type=str, default='ct', choices=["ct", "cbct"], help="the output directory")

    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")

    main(device, args)
