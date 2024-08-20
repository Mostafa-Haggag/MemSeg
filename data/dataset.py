import cv2
import os
import numpy as np
from PIL import Image
from glob import glob
from einops import rearrange

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import imgaug.augmenters as iaa

from data import rand_perlin_2d_np

from typing import List

import cv2
import os
import numpy as np
from glob import glob
from einops import rearrange

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import imgaug.augmenters as iaa

from .perlin import rand_perlin_2d_np

from typing import List

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MemSegDataset(Dataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool, to_memory: bool = False, 
        resize: List[int] = [256, 256], imagesize: int = 224,
        texture_source_dir: str = None, structure_grid_size: str = 8,
        transparency_range: List[float] = [0.15, 1.],
        perlin_scale: int = 6, min_perlin_scale: int = 0, perlin_noise_threshold: float = 0.5,
        use_mask: bool = True, bg_threshold: float = 100, bg_reverse: bool = False
    ):
        # mode
        self.is_train = is_train 
        self.to_memory = to_memory

        # load image file list
        self.datadir = datadir # D:\datasets\mvtec
        self.target = target # target the class
        # depending you are working on train or test you get all the file_list
        self.file_list = glob(os.path.join(self.datadir, self.target, 'train/*/*' if is_train else 'test/*/*'))
        
        # synthetic anomaly
        # if yo are training and you are not to memory
        # to memory is always set to False
        if self.is_train and not self.to_memory:
            # load texture image file list
            # These is the texture source file list
            self.texture_source_file_list = glob(os.path.join(texture_source_dir,'*/*')) if texture_source_dir else None
        
            # perlin noise
            self.perlin_scale = perlin_scale# 6
            self.min_perlin_scale = min_perlin_scale # 0
            self.perlin_noise_threshold = perlin_noise_threshold # you have to threshold the noise
            
            # structure
            self.structure_grid_size = structure_grid_size # you set it here to 8 by 8 althought this
            # is different than the results of the paper
            
            # anomaly mixing
            # you have a range of 2 values
            self.transparency_range = transparency_range
            
            # mask setting
            # these are set for you from the anomaley_mask.json remember this
            self.use_mask = use_mask
            self.bg_threshold = bg_threshold
            self.bg_reverse = bg_reverse
            
        # transform
        self.resize = list(resize)
        # you resize the imagei nto 288 by 288
        # you transofmr to 256 by 256 for the network
        self.transform_img = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        # very important to use image net mean and std
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]# image size beta3ak is 256 by 256
        self.transform_mask = transforms.Compose(self.transform_mask)

        # sythetic anomaly switch
        self.anomaly_switch = False
        
    def __getitem__(self, idx):
        
        file_path = self.file_list[idx]
        
        # image
        # opening with pillow
        img = Image.open(file_path).convert("RGB").resize(self.resize)
        # converting to numpy arrray
        img = np.array(img)
        
        # target
        # this is my target label
        target = 0 if 'good' in self.file_list[idx] else 1
        
        # mask
        if 'good' in file_path:
            mask = np.zeros(self.resize, dtype=np.float32)
        else:
            # getting the mask this will help us understand where is the problem
            # if i understand correctly in the
            mask = Image.open(file_path.replace('test','ground_truth').replace('.png','_mask.png')).resize(self.resize)
            mask = np.array(mask)
        ## anomaly source
        if self.is_train and not self.to_memory:
            if self.anomaly_switch:
                img, mask = self.generate_anomaly(img=img, texture_img_list=self.texture_source_file_list)
                # you took a good sample and you made it bad
                target = 1
                self.anomaly_switch = False
                mask = torch.Tensor(mask)
            else:
                # turn the switch to true
                self.anomaly_switch = True
        # in this way we have 50 percent of the times we have created annomlies
        img = self.transform_img(img)
        mask = self.transform_mask(mask).squeeze()
        return img, mask, target
        
        
    def rand_augment(self):
        augmenters = [
            iaa.GammaContrast((0.5,2.0),per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50,50),per_channel=True),
            iaa.Solarize(0.5, threshold=(32,128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
        aug = iaa.Sequential([
            augmenters[aug_idx[0]],
            augmenters[aug_idx[1]],
            augmenters[aug_idx[2]]
        ])
        
        return aug
        
    def generate_anomaly(self, img: np.ndarray, texture_img_list: list = None) -> List[np.ndarray]:
        '''
        step 1. generate mask
            - target foreground mask
            - perlin noise mask
            
        step 2. generate texture or structure anomaly
            - texture: load DTD
            - structure: we first perform random adjustment of mirror symmetry, rotation, brightness, saturation, 
            and hue on the input image  𝐼 . Then the preliminary processed image is uniformly divided into a 4×8 grid 
            and randomly arranged to obtain the disordered image  𝐼 
            
        step 3. blending image and anomaly source
        '''
        
        # step 1. generate mask
        img_size = img.shape[:-1] # H x W
        
        ## target foreground mask
        # this is very imporatnt that in anomley_mask
        # you have option to use the mask and not to use it
        # also if you use it you have the option to choose threshold value

        if self.use_mask:
            target_foreground_mask = self.generate_target_foreground_mask(img=img)
        else:
            target_foreground_mask = np.ones(self.resize)
        
        ## perlin noise mask
        perlin_noise_mask = self.generate_perlin_noise_mask(img_size=img_size)
        
        ## mask
        mask = perlin_noise_mask * target_foreground_mask
        # The function np.expand_dims() is used
        # to add an additional dimension to an array in a specific axis. # new size (height, width, 1).
        mask_expanded = np.expand_dims(mask, axis=2)
        
        # step 2. generate texture or structure anomaly
        
        ## anomaly source
        anomaly_source_img = self.anomaly_source(img=img, texture_img_list=texture_img_list)
        
        ## mask anomaly parts
        factor = np.random.uniform(*self.transparency_range, size=1)[0]
        anomaly_source_img = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)
        
        # step 3. blending image and anomaly source
        anomaly_source_img = ((- mask_expanded + 1) * img) + anomaly_source_img
        
        return (anomaly_source_img.astype(np.uint8), mask)
    
    def generate_target_foreground_mask(self, img: np.ndarray) -> np.ndarray:
        # convert RGB into GRAY scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # generate binary mask of gray scale image
        _, target_background_mask = cv2.threshold(img_gray, self.bg_threshold, 255, cv2.THRESH_BINARY)
        # The binary mask image where the background pixels are white (255) and the rest are black (0).
        target_background_mask = target_background_mask.astype(np.bool_).astype(int)

        # invert mask for foreground mask
        if self.bg_reverse:
            target_foreground_mask = target_background_mask
            # fulll background is 1 and foreground is 0
        else:
            target_foreground_mask = -(target_background_mask - 1)
            #  This effectively flips the mask to create a foreground mask where the foreground
            #  areas are marked as 1 and the background areas are marked as 0.
            # the background is zero and foreground is 1
        
        return target_foreground_mask
    
    def generate_perlin_noise_mask(self, img_size: tuple) -> np.ndarray:
        # define perlin noise scale
        # # torch.randint(min_perlin_scale, perlin_scale, (1,)) gives you valuees between min perlin scale ot perlin scale
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])

        # generate perlin noise        
        perlin_noise = rand_perlin_2d_np(img_size, (perlin_scalex, perlin_scaley))
        
        # apply affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)
        
        # make a mask by applying threshold
        # this is how to make thresholding on the noisse
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold, 
            np.ones_like(perlin_noise), 
            np.zeros_like(perlin_noise)
        )
        
        return mask_noise
    
    def anomaly_source(self, img: np.ndarray, texture_img_list: list = None) -> np.ndarray:
        # choose randomly the source of the anomaly whether strucutalre or texture
        p = np.random.uniform() if texture_img_list else 1.0
        if p < 0.5:
            idx = np.random.choice(len(texture_img_list))
            # choose a random one
            img_size = img.shape[:-1] # H x W
            anomaly_source_img = self._texture_source(img_size=img_size, texture_img_path=texture_img_list[idx])
        else:
            anomaly_source_img = self._structure_source(img=img)
            
        return anomaly_source_img
        
    def _texture_source(self, img_size: tuple, texture_img_path: str) -> np.ndarray:
        texture_source_img = cv2.imread(texture_img_path)
        texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
        texture_source_img = cv2.resize(texture_source_img, dsize=img_size).astype(np.float32)
        
        return texture_source_img
        
    def _structure_source(self, img: np.ndarray) -> np.ndarray:
        # #First Rearrangement: Converts the tensor from a grid-based layout to
        # a flattened grid layout where images are ordered sequentially.
        # #Shuffling: Randomizes the order of the images to ensure varied input.
        # #Second Rearrangement: Converts the tensor back to its original
        # grid-based layout but with the images shuffled.
        # VIP INFO
        # input  (h w) gw gh c where h and w are now dimensions of the flattened grid, and gw, gh, and c are preserved.
        # you are rturning image to as it was after shuffeling is by the disorder idx
        # output is : (h gh) (w gw) c where the grid dimensions are restored to their original structure.
        structure_source_img = self.rand_augment()(image=img)
        
        img_size = img.shape[:-1] # H x W
        
        assert img_size[0] % self.structure_grid_size == 0, 'structure should be devided by grid size accurately'
        grid_w = img_size[1] // self.structure_grid_size
        grid_h = img_size[0] // self.structure_grid_size
        # # h and w are the height and width of the grid.
        # # gh and gw are the height and width of the grid cells.
        structure_source_img = rearrange(
            tensor  = structure_source_img, 
            pattern = '(h gh) (w gw) c -> (h w) gw gh c',
            gw      = grid_w, 
            gh      = grid_h
        )
        disordered_idx = np.arange(structure_source_img.shape[0])
        np.random.shuffle(disordered_idx)

        structure_source_img = rearrange(
            tensor  = structure_source_img[disordered_idx], 
            pattern = '(h w) gw gh c -> (h gh) (w gw) c',
            h       = self.structure_grid_size,
            w       = self.structure_grid_size
        ).astype(np.float32)
        
        return structure_source_img
        
    def __len__(self):
        return len(self.file_list)
    

