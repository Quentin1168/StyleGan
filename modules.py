from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

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
    

class z_to_w(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.z_to_w = nn.Sequential(FCLayer(in_ch, out_ch),
                                    nn.ReLU(),
                                    FCLayer(in_ch, out_ch),
                                    nn.ReLU(),
                                    FCLayer(in_ch, out_ch),
                                    nn.ReLU(),
                                    FCLayer(in_ch, out_ch),
                                    nn.ReLU(),
                                    FCLayer(in_ch, out_ch),
                                    nn.ReLU(),
                                    FCLayer(in_ch, out_ch),
                                    nn.ReLU(),
                                    FCLayer(in_ch, out_ch),
                                    nn.ReLU(),)
                                    
        
    def forward(self, x):
        return self.z_to_w(x)
    
class A(nn.Module):
    
    def __init__(self, in_ch, out_ch, hidn_ch):
        super().__init__()
        
        self.scale = nn.Linear(in_ch, hidn_ch)
        self.shift = nn.Linear(in_ch, out_ch)
        
    def forward(self, x):
        scale = self.scale(x)
        shift = self.shift(x)
        
        return scale, shift

class FCLayer(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.FC = nn.Linear(in_ch, out_ch)
        
        #set normal distribution N(0, 1)
        self.FC.weight.data.normal_()
        self.FC.bias.data.zero_()
        
    def forward(self, x):
        return self.FC(x)


class SynthLayer(nn.Module):
    
    def __init__(self, in_ch, out_ch, hidn_ch, lat_in_ch, lat_out_ch, scale = True, first = False):
        super().__init__()
        self.const = torch.zeros(in_ch)
        conv_list = []
        self.first = first
   
        self.lat_in_ch = lat_in_ch
       
        self.conv1 = nn.Conv2d(in_ch, hidn_ch, kernel_size=3, 
                           stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidn_ch, out_ch, kernel_size=3,
                           stride=1, padding=1)

        self.scale = scale
        self.gen_noise = torch.randn_like
        self.upsample = F.interpolate #learnt upsample
        self.noise_factor = 0.01
        
        self.adaIN1 = adaIN()
        self.adaIN2 = adaIN()
        self.A = A(in_ch, out_ch, hidn_ch)
        self.z_w = z_to_w(lat_in_ch, lat_out_ch)
        
    def forward(self, z, x):
        if self.first = False:
            w = self.z_w(z)
            w = self.A(w)
            x = self.conv1(x)
            x = x + self.gen_noise
            x = self.adaIN1(x,w)
            
            x = self.conv2(x)
            x = x + self.gen_noise
            x = self.adaIN2(x,w)
        
        else:
            w = self.z_w(z)
            w = self.A(w)
            x = x + self.gen_noise
            x = self.adaIN1(x,w)
            x = self.conv1(x)
            x = x + self.gen_noise
            x = self.adaIN2(x, w)
        return x
        