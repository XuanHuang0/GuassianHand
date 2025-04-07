# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import cv2
import lpips
import numpy as np
from skimage.metrics import structural_similarity

class Evaluator:
    def __init__(self):
        self.result_dir = None
        self.use_gpu = True
        self.loss_fn = lpips.LPIPS(net='alex')
        if self.use_gpu:
            self.loss_fn.cuda()
    
    @staticmethod
    def _compute_psnr(img_pred, img_gt):
        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def _compute_ssim(self, rgb_pred, rgb_gt, input_imgs, mask_at_box, human_idx, frame_index, view_index):
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        img_pred = rgb_pred[y:y + h, x:x + w]
        img_gt = rgb_gt[y:y + h, x:x + w]

        human_dir = os.path.join(self.result_dir, human_idx)
        pred_dir = os.path.join(human_dir, 'pred')
        gt_dir = os.path.join(human_dir, 'gt')
        input_dir = os.path.join(human_dir, 'input')

        for _p in [pred_dir, gt_dir, input_dir]:
            os.system(f'mkdir -p {_p}')
        cv2.imwrite(os.path.join(gt_dir, f'frame{frame_index}_view{view_index}_gt.png'), (img_gt[..., [2, 1, 0]]*255))
        cv2.imwrite(os.path.join(pred_dir, f'frame{frame_index}_view{view_index}.png'), (img_pred[..., [2, 1, 0]]*255))
        
        input_imgs = (input_imgs[..., [2, 1, 0]] * 255.).astype(np.uint8)
        for view in range(input_imgs.shape[0]):
            cv2.imwrite(os.path.join(input_dir, f'frame{frame_index}_t_0_view_{view_index}.png'), input_imgs[view][y:y + h, x:x + w])

        ssim = structural_similarity(img_pred, img_gt, multichannel=True)
        return ssim

    def _compute_lpips(self, rgb_pred, rgb_gt, input_imgs, mask_at_box, human_idx, frame_index, view_index):
        
        human_dir = os.path.join(self.result_dir, human_idx)
        pred_dir = os.path.join(human_dir, 'pred')
        gt_dir = os.path.join(human_dir, 'gt')
        input_dir = os.path.join(human_dir, 'input')

        gt_img_path=os.path.join(gt_dir, f'frame{frame_index}_view{view_index}_gt.png')
        pred_img_path=os.path.join(pred_dir, f'frame{frame_index}_view{view_index}.png')
        
        img0 = lpips.im2tensor(lpips.load_image(gt_img_path))  # RGB image from [-1,1]
        img1 = lpips.im2tensor(lpips.load_image(pred_img_path))

        if self.use_gpu:
            img0 = img0.cuda()
            img1 = img1.cuda()
        dist01 = self.loss_fn.forward(img0, img1.to(img0.device))
        return dist01

    def draw_keypoints(self, img, kpts, color=(255, 0, 0), size=3):
        for i in range(kpts.shape[0]):
            kp2 = kpts[i].tolist()
            kp2 = [int(kp2[0]), int(kp2[1])]
            img = cv2.circle(img, kp2, 2, color, size)
        return img

    def draw_keypoints_vis(self, img, kpts,vis, color=(255, 0, 0), size=3):
        vis=vis.squeeze(-1)
        for i in range(kpts.shape[0]):
            kp2 = kpts[i].tolist()
            kp2 = [int(kp2[0]), int(kp2[1])]
            if int(vis[i]) == 1:
                img = cv2.circle(img, kp2, 2, color, size)
            else:
                img = cv2.circle(img, kp2, 2, (0, 255, 0), size)
        return img

    def compute_score(self, rgb_pred, rgb_gt, input_imgs, mask_at_box, human_idx, frame_index, view_index, ka_xy=None,vert_vis=None, vert_xy=None, joint_xy=None, fake_vis_pred=None, real_vis_pred=None, vis_img_gt=None, msk=None):
        """ Compute MSE, PNSR, and SSIM. 
        Args:
            rgb_pred (torch.tensor): (1, 3, H, W) [0,1]
            rgb_gt (torch.tensor): (1, 3, H, W) [0,1]
            mask_at_box (torch.tensor): (1, H, W)
            input_imgs (torch.tensor): (V, 3, H, W) [0,1]
            human_idx (str): id of the human 
        """
        rgb_pred = rgb_pred.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        input_imgs = input_imgs.detach().cpu().numpy()  # V, H, W, 1
        rgb_gt = rgb_gt.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        mask_at_box = mask_at_box.squeeze().detach().cpu().numpy()  # H, W

        human_dir = os.path.join(self.result_dir, human_idx)
        pred_dir = os.path.join(human_dir, 'pred')
        gt_dir = os.path.join(human_dir, 'gt')
        input_dir = os.path.join(human_dir, 'input')
        for _p in [pred_dir, gt_dir, input_dir]:
            os.system(f'mkdir -p {_p}')
        
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))

        mse = np.mean((rgb_pred - rgb_gt) ** 2)
        psnr = self._compute_psnr(rgb_pred, rgb_gt)
        ssim = self._compute_ssim(rgb_pred, rgb_gt, input_imgs, mask_at_box, human_idx, frame_index, view_index)
        lpips = self._compute_lpips(rgb_pred, rgb_gt, input_imgs, mask_at_box, human_idx, frame_index, view_index)

        return {
            'mse': float(mse), 
            'psnr': float(psnr),
            'ssim': float(ssim),
            'lpips':float(lpips),
        }
