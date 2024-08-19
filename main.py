import wandb
import logging
import os
import torch
import torch.nn as nn
import argparse

from omegaconf import OmegaConf
from timm import create_model
from data import create_dataset, create_dataloader
from models import MemSeg, MemoryBank
from focal_loss import FocalLoss
from train import training
from log import setup_default_logging
from utils import torch_seed
from scheduler import CosineAnnealingWarmupRestarts


_logger = logging.getLogger('train')


def run(cfg):

    # setting seed and device
    setup_default_logging()
    torch_seed(cfg.SEED)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # getting the device
    _logger.info('Device: {}'.format(device))
    # logging the device

    # savedir
    cfg.EXP_NAME = cfg.EXP_NAME + f"-{cfg.DATASET.target}"
    # the experiment name usualy it is Memseg
    # target is the called
    # you call the experiment Memseg-Target_category
    # you save everything under savdir
    # there is path called saved_model
    savedir = os.path.join(cfg.RESULT.savedir, cfg.EXP_NAME)
    os.makedirs(savedir, exist_ok=True)

    
    # wandb
    if cfg.TRAIN.use_wandb:
        wandb.init(name=cfg.EXP_NAME, project='MemSeg', config=OmegaConf.to_container(cfg))

    # build datasets
    trainset = create_dataset(
        datadir                = cfg.DATASET.datadir,
        target                 = cfg.DATASET.target, 
        is_train               = True,
        resize                 = cfg.DATASET.resize,
        imagesize              = cfg.DATASET.imagesize,
        texture_source_dir     = cfg.DATASET.texture_source_dir,
        structure_grid_size    = cfg.DATASET.structure_grid_size,
        transparency_range     = cfg.DATASET.transparency_range,
        perlin_scale           = cfg.DATASET.perlin_scale,
        min_perlin_scale       = cfg.DATASET.min_perlin_scale,
        perlin_noise_threshold = cfg.DATASET.perlin_noise_threshold,
        use_mask               = cfg.DATASET.use_mask,
        bg_threshold           = cfg.DATASET.bg_threshold,
        bg_reverse             = cfg.DATASET.bg_reverse
    )
    # this is a memory dataset that will be kept safe in the memory of the algorithim
    memoryset = create_dataset(
        datadir   = cfg.DATASET.datadir,
        target    = cfg.DATASET.target, 
        is_train  = True,
        to_memory = True,
        resize    = cfg.DATASET.resize,
        imagesize = cfg.DATASET.imagesize,
    )

    testset = create_dataset(
        datadir   = cfg.DATASET.datadir,
        target    = cfg.DATASET.target, 
        is_train  = False,
        resize    = cfg.DATASET.resize,
        imagesize = cfg.DATASET.imagesize,
    )
    
    # build dataloader
    trainloader = create_dataloader(
        dataset     = trainset,
        train       = True,
        batch_size  = cfg.DATALOADER.batch_size,
        num_workers = cfg.DATALOADER.num_workers
    )
    
    testloader = create_dataloader(
        dataset     = testset,
        train       = False,
        batch_size  = cfg.DATALOADER.batch_size,
        num_workers = cfg.DATALOADER.num_workers
    )


    # build feature extractor
    feature_extractor = create_model(
        cfg.MODEL.feature_extractor_name, 
        pretrained    = True, 
        features_only = True
    ).to(device)
    ## freeze weight of layer1,2,3
    for l in ['layer1','layer2','layer3']:
        for p in feature_extractor[l].parameters():
            p.requires_grad = False

    # build memory bank
    memory_bank = MemoryBank(
        normal_dataset   = memoryset,
        nb_memory_sample = cfg.MEMORYBANK.nb_memory_sample,
        device           = device
    )
    ## update normal samples and save
    memory_bank.update(feature_extractor=feature_extractor)
    torch.save(memory_bank, os.path.join(savedir, f'memory_bank.pt'))
    _logger.info('Update {} normal samples in memory bank'.format(cfg.MEMORYBANK.nb_memory_sample))

    # build MemSeg
    model = MemSeg(
        memory_bank       = memory_bank,
        feature_extractor = feature_extractor
    ).to(device)

    # Set training
    l1_criterion = nn.L1Loss()
    f_criterion = FocalLoss(
        gamma = cfg.TRAIN.focal_gamma, 
        alpha = cfg.TRAIN.focal_alpha
    )

    optimizer = torch.optim.AdamW(
        params       = filter(lambda p: p.requires_grad, model.parameters()), 
        lr           = cfg.OPTIMIZER.lr, 
        weight_decay = cfg.OPTIMIZER.weight_decay
    )

    if cfg['SCHEDULER']['use_scheduler']:
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, 
            first_cycle_steps = cfg.TRAIN.num_training_steps,
            max_lr = cfg.OPTIMIZER.lr,
            min_lr = cfg.SCHEDULER.min_lr,
            warmup_steps   = int(cfg.TRAIN.num_training_steps * cfg.SCHEDULER.warmup_ratio)
        )
    else:
        scheduler = None

    # Fitting model
    training(
        model              = model, 
        num_training_steps = cfg.TRAIN.num_training_steps, 
        trainloader        = trainloader, 
        validloader        = testloader, 
        criterion          = [l1_criterion, f_criterion], 
        loss_weights       = [cfg.TRAIN.l1_weight, cfg.TRAIN.focal_weight],
        optimizer          = optimizer,
        scheduler          = scheduler,
        log_interval       = cfg.LOG.log_interval,
        eval_interval      = cfg.LOG.eval_interval,
        savedir            = savedir,
        device             = device,
        use_wandb          = cfg.TRAIN.use_wandb
    )



if __name__=='__main__':
    # you get it from the running place
    # This line loads command-line arguments using OmegaConf.from_cli(),
    # which parses the command-line inputs into a dictionary-like object (args).
    # The keys and values correspond to the command-line options passed when running the script.
    # The command used to run is
    # python main.py configs=configs.yaml DATASET.target=bottle
    # like this you are setting a specific thing
    args = OmegaConf.from_cli()


    # load default config
    # i am loading the yaml file
    cfg = OmegaConf.load(args.configs)
    del args['configs']
    
    # merge config with new keys
    # i am merging both so that  DATASET.target=bottle gets set to bottle
    cfg = OmegaConf.merge(cfg, args)
    
    # target cfg
    # ./anomaly_mask.json
    # what is this file? i am doing to load it and merge it with yaml to have everything complete
    target_cfg = OmegaConf.load(cfg.DATASET.anomaly_mask_info)
    cfg.DATASET = OmegaConf.merge(cfg.DATASET, target_cfg[cfg.DATASET.target])
    
    print(OmegaConf.to_yaml(cfg))

    run(cfg)
