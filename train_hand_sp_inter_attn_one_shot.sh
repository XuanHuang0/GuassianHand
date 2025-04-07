CUDA_VISIBLE_DEVICES=2 python infer_one_shot.py --config ./config/config_one_shot.yaml --config_hand ./config/one_shot.json
CUDA_VISIBLE_DEVICES=4 python infer_one_shot.py --config ./config/config_one_shot.yaml --config_hand ./config/one_shot.json --run_val

CUDA_VISIBLE_DEVICES=4 python infer_one_shot_edit.py --config ./config/config_one_shot_edit.yaml --config_hand ./config/one_shot_edit.json
CUDA_VISIBLE_DEVICES=2 python infer_one_shot_edit.py --config ./config/config_one_shot_edit_drive.yaml --config_hand ./config/one_shot_edit.json --run_val

CUDA_VISIBLE_DEVICES=3 python infer_one_shot_edit.py --config ./config/config_one_shot_edit.yaml --config_hand ./config/one_shot_avatar.json
CUDA_VISIBLE_DEVICES=3 python infer_one_shot_edit.py --config ./config/config_one_shot_avatar_drive.yaml --config_hand ./config/one_shot_avatar.json --run_val