import torch
import numpy as np
import os
import torch.nn.functional as F
from models import RGB2YCrCb, YCrCb2RGB

from PIL import Image


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader):
        image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)
        with torch.no_grad():
            for i, (x, y, irImage, visImage) in enumerate(val_loader):
                x_cond = x[:, :2, :, :].to(self.diffusion.device)

                b, c, h, w = x_cond.shape
                img_h_32 = int(32 * np.ceil(h / 32.0))
                img_w_32 = int(32 * np.ceil(w / 32.0))
                x_cond = F.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
                x_output = self.diffusive_restoration(x_cond)
                x_output = x_output[:, :, :h, :w]
                fuseImageY = torch.max(x_output, dim=1, keepdim=True)[0]

                images_vis_ycrcb = RGB2YCrCb(visImage)

                fusion_ycrcb = torch.cat(
                    (fuseImageY, images_vis_ycrcb[:, 1:2, :,
                                 :], images_vis_ycrcb[:, 2:, :, :]),
                    dim=1,
                )

                fusion_image = YCrCb2RGB(fusion_ycrcb)

                ones = torch.ones_like(fusion_image)
                zeros = torch.zeros_like(fusion_image)
                fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
                fusion_image = torch.where(
                    fusion_image < zeros, zeros, fusion_image)
                fused_image = fusion_image.cpu().numpy()
                fused_image = fused_image.transpose((0, 2, 3, 1))
                fused_image = (fused_image - np.min(fused_image)) / (
                        np.max(fused_image) - np.min(fused_image)
                )
                fused_image = np.uint8(255.0 * fused_image)

                for k in range(b):
                    image = fused_image[k, :, :, :]
                    image = image.squeeze()
                    image = Image.fromarray(image)
                    save_path = os.path.join(image_folder, f"{y[0]}.png")
                    image.save(save_path)
                    print('Fusion {0} Sucessfully!'.format(save_path))

    def diffusive_restoration(self, x_cond):
        x_output = self.diffusion.model(x_cond)
        return x_output["pred_x"]
