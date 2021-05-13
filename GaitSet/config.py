conf_CASIA = {
    "WORK_PATH": "./outputs/CASIA-B_SetNet_OU",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "data": {
        'dataset_path': "./data/CASIA-B",
        'resolution': '64',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: GaitSet.model.utils.data_loader.load_data
        'pid_num': 73,
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (8, 16),
        'restore_iter': 0,
        'total_iter': 100000,
        'margin': 0.2,
        'num_workers': 4,
        'frame_num': 30,
        'model_name': 'GaitSet',
    },
}

conf_OULP = {
    "WORK_PATH": "./outputs/OULP_SetNet_OU",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "data": {
        'dataset_path': "./data/OULP",
        'resolution': '64',
        'dataset': 'OU-LP',
        # For more detail, please refer to
        # function: GaitSet.model.utils.data_loader.load_data
        'pid_num': 3836,
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (16, 8),
        'restore_iter': 0,
        'total_iter': 100000,
        'margin': 0.2,
        'num_workers': 4,
        'frame_num': 30,
        'model_name': 'GaitSet',
    },
}
