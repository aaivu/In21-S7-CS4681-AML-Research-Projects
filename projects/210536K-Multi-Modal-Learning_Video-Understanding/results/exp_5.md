SGP backbone + identity neck + normal head (30 epochs + 5 warm up)

{'dataset': {'crop_ratio': [0.9, 1.0],
             'default_fps': None,
             'downsample_rate': 1,
             'feat_folder': './data/thumos/i3d_features',
             'feat_stride': 4,
             'file_ext': '.npy',
             'file_prefix': None,
             'force_upsampling': False,
             'input_dim': 2048,
             'json_file': './data/thumos/annotations/thumos14.json',
             'max_seq_len': 2304,
             'num_classes': 20,
             'num_frames': 16,
             'trunc_thresh': 0.5},
 'dataset_name': 'thumos',
 'devices': ['cuda:0'],
 'init_rand_seed': 1234567891,
 'loader': {'batch_size': 2, 'num_workers': 4},
 'model': {'backbone_arch': [2, 2, 5],
           'backbone_type': 'SGP',
           'boudary_kernel_size': 3,
           'downsample_type': 'max',
           'embd_dim': 512,
           'embd_kernel_size': 3,
           'embd_with_ln': True,
           'fpn_dim': 512,
           'fpn_type': 'identity',
           'fpn_with_ln': True,
           'head_dim': 512,
           'head_kernel_size': 3,
           'head_num_layers': 3,
           'head_with_ln': True,
           'init_conv_vars': 0,
           'input_dim': 2048,
           'input_noise': 0.0005,
           'iou_weight_power': 0.2,
           'k': 5,
           'max_buffer_len_factor': 6.0,
           'max_seq_len': 2304,
           'n_sgp_win_size': 1,
           'num_bins': 16,
           'num_classes': 20,
           'regression_range': [[0, 4],
                                [4, 8],
                                [8, 16],
                                [16, 32],
                                [32, 64],
                                [64, 10000]],
           'scale_factor': 2,
           'sgp_mlp_dim': 768,
           'test_cfg': {'duration_thresh': 0.05,
                        'ext_score_file': None,
                        'iou_threshold': 0.1,
                        'max_seg_num': 2000,
                        'min_score': 0.001,
                        'multiclass_nms': True,
                        'nms_method': 'soft',
                        'nms_sigma': 0.5,
                        'pre_nms_thresh': 0.001,
                        'pre_nms_topk': 2000,
                        'voting_thresh': 0.7},
           'train_cfg': {'center_sample': 'radius',
                         'center_sample_radius': 1.5,
                         'clip_grad_l2norm': 1.0,
                         'cls_prior_prob': 0.01,
                         'dropout': 0.0,
                         'droppath': 0.1,
                         'head_empty_cls': [],
                         'init_loss_norm': 100,
                         'label_smoothing': 0.0,
                         'loss_weight': 1.0},
           'use_abs_pe': False,
           'use_trident_head': False},
 'model_name': 'TriDet',
 'opt': {'epochs': 20,
         'eta_min': 1e-08,
         'learning_rate': 0.0001,
         'momentum': 0.9,
         'schedule_gamma': 0.1,
         'schedule_steps': [],
         'schedule_type': 'cosine',
         'type': 'AdamW',
         'warmup': True,
         'warmup_epochs': 20,
         'weight_decay': 0.025},
 'output_folder': './ckpt/',
 'test_cfg': {'duration_thresh': 0.05,
              'ext_score_file': None,
              'iou_threshold': 0.1,
              'max_seg_num': 2000,
              'min_score': 0.001,
              'multiclass_nms': True,
              'nms_method': 'soft',
              'nms_sigma': 0.5,
              'pre_nms_thresh': 0.001,
              'pre_nms_topk': 2000,
              'voting_thresh': 0.7},
 'train_cfg': {'center_sample': 'radius',
               'center_sample_radius': 1.5,
               'clip_grad_l2norm': 1.0,
               'cls_prior_prob': 0.01,
               'dropout': 0.0,
               'droppath': 0.1,
               'head_empty_cls': [],
               'init_loss_norm': 100,
               'label_smoothing': 0.0,
               'loss_weight': 1.0},
 'train_split': ['validation'],
 'val_split': ['test']}
=> loading checkpoint '/kaggle/input/epoch-35-sgp-identity-normal-head/epoch_035.pth.tar'
Loading from EMA model ...

Start testing model TriDet ...
Test: [00010/00212]	Time 0.13 (0.13)
Test: [00020/00212]	Time 0.08 (0.11)
Test: [00030/00212]	Time 0.09 (0.10)
Test: [00040/00212]	Time 0.09 (0.10)
Test: [00050/00212]	Time 0.09 (0.10)
Test: [00060/00212]	Time 0.07 (0.09)
Test: [00070/00212]	Time 0.07 (0.09)
Test: [00080/00212]	Time 0.08 (0.09)
Test: [00090/00212]	Time 0.09 (0.09)
Test: [00100/00212]	Time 0.09 (0.09)
Test: [00110/00212]	Time 0.11 (0.09)
Test: [00120/00212]	Time 0.08 (0.09)
Test: [00130/00212]	Time 0.09 (0.09)
Test: [00140/00212]	Time 0.09 (0.09)
Test: [00150/00212]	Time 0.08 (0.09)
Test: [00160/00212]	Time 0.11 (0.09)
Test: [00170/00212]	Time 0.13 (0.09)
Test: [00180/00212]	Time 0.08 (0.09)
Test: [00190/00212]	Time 0.10 (0.09)
Test: [00200/00212]	Time 0.08 (0.09)
Test: [00210/00212]	Time 0.08 (0.09)
[RESULTS] Action detection results on thumos14.

|tIoU = 0.30: mAP = 82.61 (%)
|tIoU = 0.40: mAP = 78.48 (%)
|tIoU = 0.50: mAP = 71.55 (%)
|tIoU = 0.60: mAP = 60.00 (%)
|tIoU = 0.70: mAP = 44.54 (%)
Avearge mAP: 67.44 (%)
All done! Total time: 83.84 sec

