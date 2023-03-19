# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import logging
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from tensorfn import load_config
import numpy as np
from config import DiffusionConfig
import torch.distributed as dist
import os, glob, cv2, time, shutil
from models.unet_autoenc import BeatGANsAutoencConfig
from diffusion import create_gaussian_diffusion, make_beta_schedule, ddim_steps
import torchvision.transforms as transforms
import torchvision

from src import paths

logger = logging.getLogger(__name__)

CONFIG_DIR = paths.BASE_DIR / "config"
FASHION_CONF = CONFIG_DIR / "fashion.conf"
LAST_PT_FILE_PATH = paths.CHECKPOINTS_DIR / "last.pt"
NPY_FILES = paths.TARGET_POSE / "*.npy"


def get_conf():
    logger.info("Creating Beat GANs and Autoencoder config class...")
    return BeatGANsAutoencConfig(image_size=256, 
    in_channels=3+20, 
    model_channels=128, 
    out_channels=3*2,  # also learns sigma
    num_res_blocks=2, 
    num_input_res_blocks=None, 
    embed_channels=512, 
    attention_resolutions=(32, 16, 8,), 
    time_embed_channels=None, 
    dropout=0.1, 
    channel_mult=(1, 1, 2, 2, 4, 4), 
    input_channel_mult=None, 
    conv_resample=True, 
    dims=2, 
    num_classes=None, 
    use_checkpoint=False,
    num_heads=1, 
    num_head_channels=-1, 
    num_heads_upsample=-1, 
    resblock_updown=True, 
    use_new_attention_order=False, 
    resnet_two_cond=True, 
    resnet_cond_channels=None, 
    resnet_use_zero_module=True, 
    attn_checkpoint=False, 
    enc_out_channels=512, 
    enc_attn_resolutions=None, 
    enc_pool='adaptivenonzero', 
    enc_num_res_block=2, 
    enc_channel_mult=(1, 1, 2, 2, 4, 4, 4), 
    enc_grad_checkpoint=False, 
    latent_net_conf=None)

class Predictor():
    def __init__(self):
        """Load the model into memory to make running multiple predictions efficient"""
        
        logger.info("Loading DiffusionConfig from: %s", FASHION_CONF)
        #opt = Config('./config/fashion_256.yaml')
        conf = load_config(DiffusionConfig, str(FASHION_CONF), show=False)
        #val_dataset, train_dataset = deepfashion_data.get_train_val_dataloader(opt.data, labels_required = True, distributed=False)
        logger.debug("Loaded DiffusionConfig: %s", conf)

        logger.info("Loading all tensors from '%s'", LAST_PT_FILE_PATH)
        ckpt = torch.load(str(LAST_PT_FILE_PATH))
        
        self.model = get_conf().make_model()
        self.model.load_state_dict(ckpt["ema"])
        self.model = self.model.cuda()
        self.model.eval()

        self.betas = conf.diffusion.beta_schedule.make()
        self.diffusion = create_gaussian_diffusion(self.betas, predict_xstart = False)#.to(device)
        
        logger.info("Gathering all *.npy files with glob")
        self.pose_list = glob.glob(str(NPY_FILES))
        logger.info("Number of Position References (*.npy): %s", len(self.pose_list))
        logger.debug("Snippet of pose_list: %s", self.pose_list[:5])
        
        self.transforms = transforms.Compose(
            [
                transforms.Resize((256,256), interpolation=Image.BICUBIC), 
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        
        logger.debug("Transform: %s", self.transforms)
        
        
    def predict_pose(self, 
                     image: str, 
                     num_poses: int = 1, 
                     sample_algorithm: str = "ddim", 
                     nsteps: int = 100):
        """
        Run a single prediction on the model

        Args:
            image (str): Path to the input image.
            num_poses (int, optional): Number of different poses to output. Defaults to 1.
            sample_algorithm (str, optional): _description_. Defaults to "ddim".
            nsteps (int, optional): _description_. Defaults to 100.
        """

        logger.info("predict_pose(image=%s, num_poses=%s, sample_algorithm=%s, nsteps=%s)",
                     image, num_poses, sample_algorithm, nsteps)
        logger.info("Predicting %s poses for the image '%s' with the '%s' sample algorithm", 
                    num_poses, image, sample_algorithm)

        # Read the input image.
        src = Image.open(image)
        
        # (1 ,3, 256, 256)
        src_tensor: Tensor = self.transforms(src).unsqueeze(0).cuda()
        

        # Randomly selects a number of poses from the pose_list. 
        # The amount of poses selected is determined by the 'num_poses'
        # List of Tensors representing the 3D numpy arrays with values between [0, 1]
        tgt_pose = torch.stack(
            [transforms.ToTensor()(np.load(ps)).cuda() 
             for ps in np.random.choice(self.pose_list, num_poses)], 
            0
        )
        logger.info("Target Poses: { shape: %s, range: [%s, %s] }", 
                    tgt_pose.shape, tgt_pose.min(), tgt_pose.max())
        
        src_tensor = src_tensor.repeat(num_poses, 1, 1, 1)
        logger.info("Source Tensor: { shape: %s, range: [%s, %s] }", 
                     src_tensor.shape, src_tensor.min(), src_tensor.max())

        if sample_algorithm == "ddpm":
            samples = self.diffusion.p_sample_loop(self.model, x_cond = [src_tensor, tgt_pose], progress = True, cond_scale = 2)
        elif sample_algorithm == "ddim":
            noise = torch.randn(src_tensor.shape).cuda()
            seq = range(0, 1000, 1000//nsteps)
            xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src_tensor, tgt_pose])
            samples = xs[-1].cuda()

        samples_grid: Tensor = torch.cat(
            [
                src_tensor[0], 
                torch.cat([samps for samps in samples], -1)
            ], 
            -1
        )

        samples_grid = (torch.clamp(samples_grid, -1., 1.) + 1.0) / 2.0
        
        logger.info("samples_grid: { shape: %s, range: [%s, %s] }", 
                    samples_grid.shape, samples_grid.min(), samples_grid.max())
        
        pose_grid: Tensor = torch.cat([torch.zeros_like(src_tensor[0]), torch.cat([samps[:3] for samps in tgt_pose], -1)], -1)
        
        logger.debug("src_tensor[0]: { shape: %s, range: [%s, %s] }", 
                     src_tensor[0].shape, src_tensor[0].min(), src_tensor[0].max())
        
        logger.debug("Target Pose Samples:")
        for idx, samps in enumerate(tgt_pose):
            pose_image = Image.fromarray(samps[:3])
            pose_image.save("./pose_%s.png", idx)
            
            logger.debug("> %s: { shape: %s, range:[%s, %s]}", 
                         idx, samps[:3].shape, samps[:3].min(), samps[:3].max())


        logger.info("pose_grid: { shape: %s, range: [%s, %s] }", 
                     pose_grid.shape, pose_grid.min(), pose_grid.max())
        
        logger.debug("pose_grid values: %s", pose_grid)
        logger.debug("1 - pose_grid = %s", (1 - pose_grid))
        
        # Finalizing poses and images for 'output.png'
        output: Tensor = torch.cat([1-pose_grid, samples_grid], -2)
        numpy_imgs: np.ndarray = output.unsqueeze(0).permute(0,2,3,1).detach().cpu().numpy()
        fake_imgs: np.ndarray = (255*numpy_imgs).astype(np.uint8)
        
        logger.info("output: { shape: %s, range: [%s, %s] }", 
                     output.shape, output.min(), output.max())
        logger.info("numpy_imgs: { shape: %s, range: [%s, %s] }", 
                     numpy_imgs.shape, numpy_imgs.min(), numpy_imgs.max())
        logger.info("fake_imgs: { shape: %s, range: [%s, %s] }", 
                     fake_imgs.shape, fake_imgs.min(), fake_imgs.max())
        
        output_image = Image.fromarray(fake_imgs[0])
        
        logger.info("Generating Output Image with fake_imgs[0]: { shape: %s, range: [%s, %s] }", 
                    fake_imgs.shape[1:], fake_imgs.min(), fake_imgs.max())
        logger.info("Output Image: %s", output_image)
        
        logger.info("Saving output image: 'output.png'")
        output_image.save('output.png')


    def predict_appearance(
        self,
        image,
        ref_img,
        ref_mask,
        ref_pose,
        sample_algorithm='ddim',
        nsteps=100,

        ):
        """Run a single prediction on the model"""

        src = Image.open(image)
        src = self.transforms(src).unsqueeze(0).cuda()
        
        ref = Image.open(ref_img)
        ref = self.transforms(ref).unsqueeze(0).cuda()

        mask = transforms.ToTensor()(Image.open(ref_mask)).unsqueeze(0).cuda()
        pose =  transforms.ToTensor()(np.load(ref_pose)).unsqueeze(0).cuda()


        if sample_algorithm == 'ddpm':
            samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, pose, ref, mask], progress = True, cond_scale = 2)
        elif sample_algorithm == 'ddim':
            noise = torch.randn(src.shape).cuda()
            seq = range(0, 1000, 1000//nsteps)
            xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, pose, ref, mask], diffusion=self.diffusion)
            samples = xs[-1].cuda()


        samples = torch.clamp(samples, -1., 1.)

        output = (torch.cat([src, ref, mask*2-1, samples], -1) + 1.0)/2.0

        numpy_imgs = output.permute(0,2,3,1).detach().cpu().numpy()
        fake_imgs = (255*numpy_imgs).astype(np.uint8)
        Image.fromarray(fake_imgs[0]).save('output.png')


# ref_img = "data/deepfashion_256x256/target_edits/reference_img_0.png"
# ref_mask = "data/deepfashion_256x256/target_mask/lower/reference_mask_0.png"
# ref_pose = "data/deepfashion_256x256/target_pose/reference_pose_0.npy"

# obj = Predictor()

# #obj.predict_pose(image='test.jpg', num_poses=4, sample_algorithm = 'ddim',  nsteps = 50)

# #obj.predict_appearance(image='test.jpg', ref_img = ref_img, ref_mask = ref_mask, ref_pose = ref_pose, sample_algorithm = 'ddim',  nsteps = 50)
