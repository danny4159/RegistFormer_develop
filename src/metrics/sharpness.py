import cv2
import numpy as np
import torch
from torchmetrics.metric import Metric

# class SharpnessMetric(Metric):
#     def __init__(self):
#         super().__init__()

#     def forward(self, imgs):
#         assert imgs.ndim == 4, "Input must be a 4D tensor in sharpness metric"

#         # imgs is [b, c, h, w]
#         score_list = []
        
#         for img in imgs: # img -> [c, h, w]
#             img_np = img.permute(1, 2, 0).detach().cpu().numpy()
#             if img_np.shape[2] == 3:
#                 img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
#             else:
#                 img_gray = img_np
                
#             blur_map = cv2.Laplacian(img_gray, cv2.CV_64F)
#             score = np.var(blur_map)
#             score_list.append(score)
        
#         score_mean = torch.tensor(score_list).mean()

#         return score_mean
    

class SharpnessMetric(Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False)
        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, imgs):
        assert imgs.ndim == 4, "Input must be a 4D tensor in sharpness metric"

        score_list = []
        for img in imgs:  # img -> [c, h, w]
            img_np = img.permute(1, 2, 0).detach().cpu().numpy()
            if img_np.shape[2] == 3:
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_np

            img_gray = img_gray.astype(np.float64)
            
            blur_map = cv2.Laplacian(img_gray, cv2.CV_64F)
            score = np.var(blur_map)
            score_list.append(score)

        score_mean = torch.tensor(score_list, device=self.score_sum.device).mean()
        self.score_sum += score_mean
        self.total += 1

    def compute(self):
        return self.score_sum / self.total

    def reset(self):
        self.score_sum = torch.tensor(0.0)
        self.total = torch.tensor(0)