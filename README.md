# GuassianHand: Learning Interaction-aware 3D Gaussian Splatting for One-shot Hand Avatars (NeurIPS 2024)

This is an official implementation of "[Learning Interaction-aware 3D Gaussian Splatting for One-shot Hand Avatars](https://arxiv.org/pdf/2410.08840)".

<p> 
   <img src="https://github.com/XuanHuang0/GuassianHand/blob/main/assets/banner.png"/> 
</p>

## Installation

1. Set up the environment using the provided script:

   ```
   bash scripts/create_env.sh
   ```

2. Register and download [MANO](https://mano.is.tue.mpg.de/)  data. Place `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` in folder `$ROOT/smplx/models/mano`.

## Pre-trained model

Download the [pretrained model](https://drive.google.com/file/d/1lYIBK75at5s4S748V2YKHZZHtwns7BQK/view?usp=sharing) and place it in `$ROOT/EXPERIMENTS/`.

## Data preparation

We provide the processed dataset used for one-shot avatar creation [here](https://drive.google.com/file/d/1XAO7LEsZPr7unfN9eB_XmdxbjFnRgYs-/view?usp=sharing). After downloading, please extract the contents and place them in the following directory: `$ROOT/processed_dataset/`.

## One-shot avatar creation

### Evaluation on InterHand2.6M

1. Fit the model to a reference image from InterHand2.6M:

   ```
   python infer_one_shot.py --config ./config/config_one_shot.yaml --config_hand ./config/one_shot.json
   ```

2.  Evaluate and visualize the results:

   ```
   python infer_one_shot.py --config ./config/config_one_shot.yaml --config_hand ./config/one_shot.json --run_val
   ```

### Texture editing

1. Fit the model to an edited image:

   ```
   python infer_one_shot_edit.py --config ./config/config_one_shot_edit.yaml --config_hand ./config/one_shot_edit.json
   ```

2.  Render novel views and poses:

   ```
   python infer_one_shot_edit.py --config ./config/config_one_shot_edit_drive.yaml --config_hand ./config/one_shot_edit.json --run_val
   ```

### Single to interacting hands

1. Fit the model to a single-hand image:

   ```
   infer_one_shot_edit.py --config ./config/config_one_shot_edit.yaml --config_hand ./config/one_shot_avatar.json
   ```

2.  Render interacting-hand images:

   ```
   infer_one_shot_edit.py --config ./config/config_one_shot_avatar_drive.yaml --config_hand ./config/one_shot_avatar.json --run_val
   ```

## Citation

If you find this work useful, please consider citing:

   ```
@inproceedings{huang2024learning,
  title={Learning Interaction-aware 3D Gaussian Splatting for One-shot Hand Avatars},
  author={Huang, Xuan and Li, Hanhui and Liu, Wanquan and Liang, Xiaodan and Yan, Yiqiang and Cheng, Yuhao and GAO, CHENQIANG},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
   ```

## Acknowledgements

Parts of this codebase are adapted from [livehand](https://github.com/amundra15/livehand) and [TriplaneGaussian](https://github.com/VAST-AI-Research/TriplaneGaussian). We appreciate their contributions and encourage citing them where appropriate.
