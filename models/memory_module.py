import torch 
import torch.nn.functional as F

import numpy as np
from typing import List


class MemoryBank:
    def __init__(self, normal_dataset, nb_memory_sample: int = 30, device='cpu'):
        '''
        normal_dataset: this is the dataset containing the images that will be sent to memory bank
        '''
        self.device = device
        
        # memory bank
        self.memory_information = {}
        # you store in here the different level
        '''
            MI level0: torch.Size([30, 64, 56, 56]) 
            MI level1: torch.Size([30, 128, 28, 28])
            MI level2: torch.Size([30, 256, 14, 14])
        '''
        # normal dataset
        self.normal_dataset = normal_dataset
        
        # the number of samples saved in memory bank
        # i donot want to save more than 30 picture in the memory bank
        self.nb_memory_sample = nb_memory_sample

    def update(self, feature_extractor):
        # making sure that you are in inference right now
        feature_extractor.eval()
        
        # define sample index
        samples_idx = np.arange(len(self.normal_dataset))
        np.random.shuffle(samples_idx)
        
        # extract features and save features into memory bank
        with torch.no_grad():
            for i in range(self.nb_memory_sample):
                # select image
                # you donot care about image or the label
                input_normal, _, _ = self.normal_dataset[samples_idx[i]]
                # modving image to device
                input_normal = input_normal.to(self.device)
                
                # extract features
                features = feature_extractor(input_normal.unsqueeze(0))
                #  0th dimension using unsqueeze(0), converting it from shape [C, H, W] (channels, height, width)
                #  to [1, C, H, W] to add a batch dimension.

                # save features into memoery bank
                for i, features_l in enumerate(features[1:-1]):
                    #  This implies that features from the first layer and the last layer are being ignored
                    #  , possibly because they are either too low-level or too high-level for the task at hand.
                    if f'level{i}' not in self.memory_information.keys():
                        # If it doesn't exist, it creates a new entry and assigns the current features_l tensor to it.
                        self.memory_information[f'level{i}'] = features_l
                    else:
                        self.memory_information[f'level{i}'] = torch.cat([self.memory_information[f'level{i}'], features_l], dim=0)
                        # If it does exist, it concatenates the new features_l tensor with the existing
                        # one along the 0th dimension (batch dimension) using torch.cat(). This essentially
                        # accumulates features from multiple samples into a single tensor, building up the memory bank.

    def _calc_diff(self, features: List[torch.Tensor]) -> torch.Tensor:
        # batch size X the number of samples saved in memory
        # i have batch size
        diff_bank = torch.zeros(features[0].size(0), self.nb_memory_sample).to(self.device)

        # level
        for l, level in enumerate(self.memory_information.keys()):
            # batch because you have big batch so you are looping example by example
            for b_idx, features_b in enumerate(features[l]):
                # calculate l2 loss
                diff = F.mse_loss(
                    input     = torch.repeat_interleave(features_b.unsqueeze(0), repeats=self.nb_memory_sample, dim=0), 
                    target    = self.memory_information[level], 
                    reduction ='none'
                ).mean(dim=[1,2,3])

                # sum loss
                diff_bank[b_idx] += diff
                
        return diff_bank

    def select(self, features: List[torch.Tensor]) -> torch.Tensor:
        # calculate difference between features and normal features of memory bank
        diff_bank = self._calc_diff(features=features)
        
        # concatenate features with minimum difference features of memory bank
        for l, level in enumerate(self.memory_information.keys()):
            
            selected_features = torch.index_select(self.memory_information[level], dim=0, index=diff_bank.argmin(dim=1))
            diff_features = F.mse_loss(selected_features, features[l], reduction='none')
            features[l] = torch.cat([features[l], diff_features], dim=1)
            
        return features
    