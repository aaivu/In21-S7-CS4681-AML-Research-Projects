import os
import sys
import argparse
import collections
import transformers
from sacred import Experiment

import torch
import data_loader.data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import utils.visualizer as module_vis
from parse_config import ConfigParser
from trainer import Multi_Trainer_dist
from utils.util import replace_nested_dict_item
from tensorboardX import SummaryWriter

# Import enhanced components
from model.model import MultiScaleVideoEncoder
from model.loss import EgoNCEWithScheduler
from model.temporal_loss import TemporalConsistencyLoss, TemporalPairBatchSampler
from data_loader.EgoClip_EgoMCQ_dataset import EgoClip_EgoMCQ

ex = Experiment('train')

@ex.main
def run():
    logger = config.get_logger('train')
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    # TODO: improve Create identity (do nothing) visualiser?
    if config['visualizer']['type'] != "":
        visualizer = config.initialize(
            name='visualizer',
            module=module_vis,
            exp_name=config['name'],
            web_dir=config._web_log_dir
        )
    else:
        visualizer = None

    torch.cuda.set_device(args.local_rank)

    # if args.world_size > 1:
    if args.master_address != 9339:
        print("DistributedDataParallel")
        # DistributedDataParallel
        torch.distributed.init_process_group(backend='nccl',
                                                 init_method='tcp://{}:{}'.format(
                                                 args.master_address, args.master_port),
                                             rank=args.rank, world_size=args.world_size)
    device = torch.device(f'cuda:{args.local_rank}')

    if args.rank == 0:
        print('world_size', args.world_size, flush=True)
        print('local_rank: ', args.local_rank, flush=True)

    # build tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'],
                                                               TOKENIZERS_PARALLELISM=False)

    # setup data_loader instances
    data_loader, valid_data_loader = init_dataloaders(config, module_data)
    if args.rank == 0:
        print('Train dataset: ', [x.n_samples for x in data_loader], ' samples')
        print('Val dataset: ', [x.n_samples for x in valid_data_loader], ' samples')
    # build model architecture, then print to console

    model = config.initialize('arch', module_arch)

    if args.local_rank == 0:
        logger.info(model)

    # get function handles of loss and metrics
    loss = config.initialize(name="loss", module=module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', transformers, trainable_params)
    lr_scheduler = None
    if 'lr_scheduler' in config._config:
        if hasattr(transformers, config._config['lr_scheduler']['type']):
            lr_scheduler = config.initialize('lr_scheduler', transformers, optimizer)
        else:
            print('lr scheduler not found')
    if config['trainer']['neptune']:
        writer = ex
    else:
        writer = None

    if args.rank == 0:
        writer = SummaryWriter(log_dir=str(config.tf_dir))

    # Use Enhanced Trainer with all multi-scale improvements
    from trainer.enhanced_trainer_egoclip import Enhanced_Multi_Trainer_dist
    
    # Check if enhancements are enabled
    use_enhancements = config.get('use_enhancements', True)
    
    if use_enhancements and args.rank == 0:
        print("=" * 60)
        print("ENHANCED EGOCLIP TRAINING")
        print("=" * 60)
        print("✓ Multi-scale video processing (4, 8, 16 frames)")
        print("✓ Temporal consistency loss")
        print("✓ Cosine temperature scheduling") 
        print("✓ Custom temporal batch sampler")
        print("=" * 60)
    
    if use_enhancements:
        trainer = Enhanced_Multi_Trainer_dist(args, model, loss, metrics, optimizer,
                          config=config,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          lr_scheduler=lr_scheduler,
                          visualizer=visualizer,
                          writer=writer,
                          tokenizer=tokenizer,
                          max_samples_per_epoch=config['trainer']['max_samples_per_epoch'])
    else:
        # Use original trainer
        trainer = Multi_Trainer_dist(args, model, loss, metrics, optimizer,
                          config=config,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          lr_scheduler=lr_scheduler,
                          visualizer=visualizer,
                          writer=writer,
                          tokenizer=tokenizer,
                          max_samples_per_epoch=config['trainer']['max_samples_per_epoch'])

    trainer.train()


def init_dataloaders(config, module_data):
    """
    Initialize enhanced dataloaders with temporal batch sampling.
    """
    # Enhanced training dataloader with temporal batch sampler
    if "type" in config["data_loader"] and "args" in config["data_loader"]:
        # Single dataloader - create enhanced version
        dataset_cls = config['data_loader']['type']
        if dataset_cls == 'EgoClip_EgoMCQ':
            # Create enhanced dataset with multi-scale support
            train_dataset = EgoClip_EgoMCQ(config['data_loader']['args'], split_type='train')
            
            # Use temporal pair batch sampler for training
            try:
                batch_sampler = TemporalPairBatchSampler(
                    dataset=train_dataset,
                    batch_size=config['data_loader']['args'].get('batch_size', 32),
                    drop_last=True
                )
                
                data_loader = [torch.utils.data.DataLoader(
                    train_dataset,
                    batch_sampler=batch_sampler,
                    num_workers=config['data_loader']['args'].get('num_workers', 4),
                    pin_memory=True
                )]
                print("✓ Using TemporalPairBatchSampler for enhanced training")
                
            except Exception as e:
                print(f"⚠ Falling back to regular DataLoader: {e}")
                data_loader = [config.initialize("data_loader", module_data)]
        else:
            data_loader = [config.initialize("data_loader", module_data)]
        
        # Validation dataloader (regular)
        config['data_loader']['args'] = replace_nested_dict_item(config['data_loader']['args'], 'split', 'val')
        config['data_loader']['args'] = replace_nested_dict_item(config['data_loader']['args'], 'batch_size', 1)
        valid_data_loader = [config.initialize("data_loader", module_data)]
        
    elif isinstance(config["data_loader"], list):
        # Multiple dataloaders
        data_loader = []
        for idx in range(len(config['data_loader'])):
            dl_config = config['data_loader'][idx]
            if dl_config.get('type') == 'EgoClip_EgoMCQ':
                # Enhanced EgoClip loader
                train_dataset = EgoClip_EgoMCQ(dl_config['args'], split_type='train')
                try:
                    batch_sampler = TemporalPairBatchSampler(
                        dataset=train_dataset,
                        batch_size=dl_config['args'].get('batch_size', 32),
                        drop_last=True
                    )
                    enhanced_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_sampler=batch_sampler,
                        num_workers=dl_config['args'].get('num_workers', 4),
                        pin_memory=True
                    )
                    data_loader.append(enhanced_loader)
                except Exception as e:
                    print(f"⚠ Enhanced loader failed for dataset {idx}, using regular: {e}")
                    data_loader.append(config.initialize('data_loader', module_data, index=idx))
            else:
                data_loader.append(config.initialize('data_loader', module_data, index=idx))
        
        # Validation dataloaders (regular)
        new_cfg_li = []
        for dl_cfg in config['data_loader']:
            dl_cfg['args'] = replace_nested_dict_item(dl_cfg['args'], 'split', 'val')
            dl_cfg['args'] = replace_nested_dict_item(dl_cfg['args'], 'batch_size', 1)
            new_cfg_li.append(dl_cfg)
        config._config['data_loader'] = new_cfg_li
        valid_data_loader = [config.initialize('data_loader', module_data, index=idx) for idx in
                             range(len(config['data_loader']))]
    else:
        raise ValueError("Check data_loader config, not correct format.")

    return data_loader, valid_data_loader


if __name__ == '__main__':
    try:    # with ddp
        master_address = os.environ['MASTER_ADDR']
        master_port = int(os.environ['MASTER_PORT'])
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
    except:  # for debug only
        master_address = 9339
        master_port = 1
        world_size = 1
        rank = 0
        local_rank = 0

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='configs/pt/egoclip.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-o', '--observe', action='store_true',
                      help='Whether to observe (neptune)')
    args.add_argument('-l', '--launcher', choices=['none', 'pytorch'], default='none',help='job launcher')
    args.add_argument('-k', '--local_rank', type=int, default=local_rank)

    args.add_argument('-ma', '--master_address', default=master_address)
    args.add_argument('-mp', '--master_port', type=int, default=master_port)
    args.add_argument('-ws', '--world_size', type=int, default=world_size)
    args.add_argument('-rk', '--rank', type=int, default=rank)
    args.add_argument('-lr1', '--learning_rate1', type=float, default=2e-4)
    args.add_argument('-sc', '--schedule', default=[60, 80])

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
    ]
    config = ConfigParser(args, options)
    args = args.parse_args()
    ex.add_config(config._config)

    if args.rank == 0:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    print("The rank(local) of this node is {}({})".format(args.rank, args.local_rank))

    if config['trainer']['neptune']:
        # delete this error if you have added your own neptune credentials neptune.ai
        raise ValueError('Neptune credentials not set up yet.')
        ex.observers.append(NeptuneObserver(
            api_token='INSERT TOKEN',
            project_name='INSERT PROJECT NAME'))
        ex.run()
    else:
        run()