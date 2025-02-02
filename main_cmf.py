import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
import platform
import pdb
import h5py


if platform.system() == 'Windows':
    NUM_WORKERS = 0
    sys.path.append(r"E:\我的坚果云\sourcecode\python\util")
else:
    NUM_WORKERS = 12
    sys.path.append("/home/chenxu/我的坚果云/sourcecode/python/util")

import common_cmf_pt as common_cmf
import common_net_pt as common_net
import common_metrics


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        const=True,
        default=16,
        help="dataset dir",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        nargs="?",
        const=True,
        default=r"/home/chenxu/datasets/cmf",
        help="dataset dir",
    )
    parser.add_argument(
        "--modality",
        type=str,
        nargs="?",
        const=True,
        default="ct",
        choices=["ct", "mri"],
        help="modality",
    )
    parser.add_argument(
        "--do_debug",
        type=int,
        nargs="?",
        const=True,
        default=0,
        choices=[0, 1],
        help="do debug",
    )

    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class DatasetAll(torch.utils.data.Dataset):
    def __init__(self, data_dir, modality, n_slices=1, debug=0, data_augment=False):
        assert modality in ("mri", "ct")

        self.data_dir = data_dir
        self.modality = modality
        self.n_slices = n_slices
        self.debug = debug
        self.data_augment = data_augment

        self.load_data()

        transforms_options = [transforms.ToTensor(),]

        if data_augment:
            transforms_options.extend([
                transforms.RandomRotation(5, fill=-1, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomResizedCrop((self.patch_height, self.patch_width), scale=(0.8, 1.0),
                                             antialias=None, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
            ])

        self.transform = transforms.Compose(transforms_options)

    def __len__(self):
        return self.num_subjects * self.slices_per_subject

    def __getitem__(self, idx):
        subject_id = idx // self.slices_per_subject
        slice_id = idx % self.slices_per_subject

        data_f = self.data_f1
        if subject_id >= self.num_subjects1:
            subject_id -= self.num_subjects1
            data_f = self.data_f2

        image = common_cmf.pad_data(np.array(data_f[self.modality][subject_id, slice_id: slice_id + self.n_slices, :, :])).transpose((1, 2, 0))

        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        image = self.transform(image)

        ret = {
            "image": image,
        }

        return ret

    def load_data(self):
        self.data_f1 = h5py.File(os.path.join(self.data_dir, "unpaired_%s.h5" % self.modality), "r")
        self.data_f2 = h5py.File(os.path.join(self.data_dir, "paired_mri_ct.h5"), "r")
        self.num_subjects1 = self.data_f1[self.modality].shape[0]
        self.data_slices, self.patch_height, self.patch_width = common_cmf.pad_data(
            np.array(self.data_f1[self.modality][0:1, :, :, :], np.float32)).shape[1:]
        self.num_subjects2 = self.data_f2[self.modality].shape[0]
        self.num_subjects = self.num_subjects1 + self.num_subjects2

        if self.debug:
            self.num_subjects = 1

        self.slices_per_subject = self.data_slices - self.n_slices + 1
        self.patch_shape = (self.n_slices, self.patch_height, self.patch_width)


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(self.config.pretty())
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(self.lightning_config.pretty())
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")


class Validation(Callback):
    def __init__(self, data_dir, modality, debug, ckptdir, n_slices, start_epoch, lr_scheduler=True):
        self.n_slices = n_slices
        self.ckptdir = ckptdir
        self.debug = debug
        self.start_epoch = start_epoch
        self.best_psnr = 0
        self.lr_schedulers = lr_scheduler

        if modality == "mri":
            if debug:
                f = h5py.File(os.path.join(data_dir, "unpaired_mri.h5"), "r")
                self.val_data = common_cmf.pad_data(np.array(f["mri"][0:1, :, :, :]))
            else:
                self.val_data, _, _ = common_cmf.load_test_data(data_dir)
        elif modality == "ct":
            if debug:
                f = h5py.File(os.path.join(data_dir, "unpaired_ct.h5"), "r")
                self.val_data = common_cmf.pad_data(np.array(f["ct"][0:1, :, :, :]))
            else:
                _, self.val_data, _ = common_cmf.load_test_data(data_dir)
        else:
            assert 0

    def on_train_epoch_end(self, trainer, pl_module):
        if self.lr_schedulers is True:
            self.lr_schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, mode="max", patience=50, cooldown=50, min_lr=1e-6, verbose=True) for optimizer in trainer.optimizers]

        pl_module.eval()

        patch_shape = (self.n_slices, self.val_data.shape[2], self.val_data.shape[3])
        psnr_list = np.zeros((self.val_data.shape[0],), np.float32)
        with torch.no_grad():
            for i in range(self.val_data.shape[0]):
                syn_im = common_net.produce_results(pl_module.device, pl_module, [patch_shape, ], [self.val_data[i], ],
                                                    data_shape=self.val_data.shape[1:], patch_shape=patch_shape,
                                                    is_seg=False, batch_size=16)
                syn_im[self.val_data[i] <= -1.] = -1.
                psnr_list[i] = common_metrics.psnr(syn_im, self.val_data[i])

        print("Val psnr:%f/%f" % (psnr_list.mean(), psnr_list.std()))
        pl_module.train()

        cur_psnr = psnr_list.mean()
        if trainer.current_epoch >= self.start_epoch:
            if cur_psnr >= self.best_psnr:
                self.best_psnr = cur_psnr
                trainer.save_checkpoint(os.path.join(self.ckptdir, "best.ckpt"))
        else:
            trainer.save_checkpoint(os.path.join(self.ckptdir, "last.ckpt"))

        if isinstance(self.lr_schedulers, list):
            for scheduler in self.lr_schedulers:
                scheduler.step(cur_psnr)


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    print("".join("%s " % x for x in sys.argv))

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")+1]
    else:
        if opt.name:
            name = "_"+opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = ""
        nowname = now+name+opt.postfix
        #logdir = os.path.join(opt.logdir, nowname)
        logdir = os.path.join(opt.logdir, opt.name)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        #trainer_config["distributed_backend"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["distributed_backend"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        # NOTE wandb < 0.10.0 interferes with shutdown
        # wandb >= 0.10.0 seems to fix it but still interferes with pudb
        # debugging (wrongly sized pudb ui)
        # thus prefer testtube for now
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # data
        #cmf_dataset = common_cmf.Dataset(opt.data_dir, opt.modality, n_slices=config.model.params.ddconfig.in_channels, debug=opt.do_debug, data_augment=True)
        cmf_dataset = DatasetAll(opt.data_dir, opt.modality, n_slices=config.model.params.ddconfig.in_channels, debug=opt.do_debug, data_augment=True)
        data = DataLoader(cmf_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=NUM_WORKERS)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main_cmf.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            #"image_logger": {
            #    "target": "main_cmf.ImageLogger",
            #    "params": {
            #        "batch_frequency": 750,
            #        "max_images": 4,
            #        "clamp": True
            #    }
            #},
            "validation": {
                "target": "main_cmf.Validation",
                "params": {
                    "data_dir": opt.data_dir,
                    "modality": opt.modality,
                    "debug": opt.do_debug,
                    "ckptdir": ckptdir,
                    "n_slices": config.model.params.ddconfig.in_channels,
                    "start_epoch": config.model.params.lossconfig.params.disc_start // len(data) + 1
                }
            },
            "learning_rate_logger": {
                "target": "main_cmf.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    #"log_momentum": True
                }
            },
        }
        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)


        # configure learning rate
        bs, base_lr = opt.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        if "accumulate_grad_batches" in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = base_lr #accumulate_grad_batches * ngpu * bs * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        import signal
        if platform.system() == 'Windows':
            signal.signal(signal.SIGTERM, melk)
            signal.signal(signal.SIGTERM, divein)
        else:
            signal.signal(signal.SIGUSR1, melk)
            signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
