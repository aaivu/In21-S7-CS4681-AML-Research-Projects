Conv backbone + FPN1D (epoch 30 + 5 warm_ups) 



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
 'model': {'backbone_arch': (2, 2, 5),
           'backbone_type': 'conv',
           'embd_dim': 512,
           'embd_kernel_size': 3,
           'embd_with_ln': True,
           'fpn_dim': 512,
           'fpn_start_level': 0,
           'fpn_type': 'fpn',
           'fpn_with_ln': True,
           'head_dim': 512,
           'head_kernel_size': 3,
           'head_num_layers': 3,
           'head_with_ln': True,
           'input_dim': 2048,
           'max_buffer_len_factor': 6.0,
           'max_seq_len': 2304,
           'n_head': 4,
           'n_mha_win_size': 19,
           'num_classes': 20,
           'regression_range': [(0, 4),
                                (4, 8),
                                (8, 16),
                                (16, 32),
                                (32, 64),
                                (64, 10000)],
           'scale_factor': 2,
           'test_cfg': {'duration_thresh': 0.05,
                        'ext_score_file': None,
                        'iou_threshold': 0.1,
                        'max_seg_num': 200,
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
           'use_rel_pe': False},
 'model_name': 'LocPointTransformer',
 'opt': {'epochs': 30,
         'learning_rate': 0.0001,
         'momentum': 0.9,
         'schedule_gamma': 0.1,
         'schedule_steps': [],
         'schedule_type': 'cosine',
         'type': 'AdamW',
         'warmup': True,
         'warmup_epochs': 5,
         'weight_decay': 0.05},
 'output_folder': './ckpt/',
 'test_cfg': {'duration_thresh': 0.05,
              'ext_score_file': None,
              'iou_threshold': 0.1,
              'max_seg_num': 200,
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
=> loading checkpoint '/kaggle/input/thumos-i3d-epoch-35-2025-10-04/epoch_035.pth.tar'
Loading from EMA model ...

Start testing model LocPointTransformer ...
Test: [00010/00212]	Time 0.14 (0.14)
Test: [00020/00212]	Time 0.08 (0.11)
Test: [00030/00212]	Time 0.08 (0.10)
Test: [00040/00212]	Time 0.10 (0.10)
Test: [00050/00212]	Time 0.09 (0.10)
Test: [00060/00212]	Time 0.08 (0.09)
Test: [00070/00212]	Time 0.08 (0.09)
Test: [00080/00212]	Time 0.09 (0.09)
Test: [00090/00212]	Time 0.09 (0.09)
Test: [00100/00212]	Time 0.08 (0.09)
Test: [00110/00212]	Time 0.10 (0.09)
Test: [00120/00212]	Time 0.08 (0.09)
Test: [00130/00212]	Time 0.09 (0.09)
Test: [00140/00212]	Time 0.09 (0.09)
Test: [00150/00212]	Time 0.08 (0.09)
Test: [00160/00212]	Time 0.11 (0.09)
Test: [00170/00212]	Time 0.12 (0.09)
Test: [00180/00212]	Time 0.08 (0.09)
Test: [00190/00212]	Time 0.10 (0.09)
Test: [00200/00212]	Time 0.08 (0.09)
Test: [00210/00212]	Time 0.08 (0.09)
[RESULTS] Action detection results on thumos14.

|tIoU = 0.30: mAP = 72.79 (%) Recall@1x = 81.72 (%) Recall@5x = 95.32 (%) 
|tIoU = 0.40: mAP = 68.37 (%) Recall@1x = 77.28 (%) Recall@5x = 93.49 (%) 
|tIoU = 0.50: mAP = 61.00 (%) Recall@1x = 70.44 (%) Recall@5x = 89.45 (%) 
|tIoU = 0.60: mAP = 50.71 (%) Recall@1x = 60.95 (%) Recall@5x = 81.36 (%) 
|tIoU = 0.70: mAP = 36.51 (%) Recall@1x = 48.95 (%) Recall@5x = 67.27 (%) 
Average mAP: 57.88 (%)
All done! Total time: 33.97 sec


