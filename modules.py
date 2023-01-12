from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms
import torch.cuda
import torch.nn as nn


"""
Adaptive Instance Normalisation (AdaIN)

Takes in a content input x and style input y, and algins the mean and variance
of the content input to that of the style's in a per channel basis.

AdaIN is used in StyleGan due to the significant improvement in performance 
when transferring styles. 

"""
class adaIN(nn.Module):
    
    def __init__(self):
        super().__init__()
    
        
    """
    Instance Normalisation normalises independent of sample and channel.
    Based on equation 5 and 6 of the paper.
    """
    def spatial_mean(self, x):
        
        #sum the values along the height and width (spatial) dimensions
        spatial_sum = torch.sum(x, (2,3))
        
        #total size of the tensor, height * width
        denom = x.size(2) * x.size(3)
        
        return spatial_sum/denom
   
    
    def spatial_std(self, x):
       spatial_mean = self.spatial_mean(x)
       #epsilon is a hyperparamter to avoid dividing by 0
       e = 1e-9 
       #broadcast to subtract mean tensor and input
       
       #need to permute to put H, W in front to broadcast
       
       x = x.permute([2,3,0,1])
       x = x - spatial_mean
       
       #revert after broadcast
       x = x.permute([2,3,0,1])
       denom = x.size(2) * x.size(3)
       
       std = torch.sqrt((torch.sum(x, (2,3))**2 + e)/denom)
       
       return std

    def forward(self, x, y):
        return (self.spatial_std(y)*((x.permute([2,3,0,1]) 
                                      - self.spatial_mean(x))/
                                     self.spatial_std(x)) \
                + self.spatial_mean(y)).permute([2,3,0,1])
    
    
