from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F


factors = [1,1,1,1/2,1/4,1/8,1/16,1/32]


class WS(nn.Module):
    
    def __init__(self, layer):
        super(WS, self).__init__()
        
        self.layer = layer
        
        if isinstance(self.layer, nn.Conv2d):
            self.scale = (2 / (self.layer.in_channels * (self.layer.kernel_size ** 2))) ** 0.5
        else:
            self.scale = (2 / self.layer.in_channels) ** 0.5
        
        self.bias = self.layer.bias
        self.layer.bias = None
        
        nn.init.normal_(self.layer.weight)
        nn.init.zeros_(self.bias)
        

    def forward(self, x):
        
        if isinstance(self.layer, nn.Conv2d):
            return self.layer(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)
        else:
            return self.layer(x*self.scale) + self.bias
        
        
class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8
        
    def forward(self, x):
        div = torch.sqrt(torch.mean(x ** 2, dim =1, keepdim = True) + self.epsilon)
        return x / div

            
        
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
        self.pixel_norm = PixelNorm() 
        self.z_to_w = nn.Sequential(WS(nn.Linear(in_ch, out_ch)),
                                    nn.ReLU(),
                                    WS(nn.Linear(in_ch, out_ch)),
                                    nn.ReLU(),
                                    WS(nn.Linear(in_ch, out_ch)),
                                    nn.ReLU(),
                                    WS(nn.Linear(in_ch, out_ch)),
                                    nn.ReLU(),
                                    WS(nn.Linear(in_ch, out_ch)),
                                    nn.ReLU(),
                                    WS(nn.Linear(in_ch, out_ch)),
                                    nn.ReLU(),
                                    WS(nn.Linear(in_ch, out_ch)),
                                    nn.ReLU())
                                    
        
    def forward(self, x):
        normalized = self.pixel_norm(x)
        return self.z_to_w(normalized)


class SynthLayer(nn.Module):
    
    def __init__(self, in_ch, out_ch, hidn_ch, scale = True, first = False):
        super().__init__()
        self.out_ch = out_ch
        self.first = first
              
        self.conv1 = WS(nn.Conv2d(in_ch, hidn_ch, kernel_size=3, 
                           stride=1, padding=1))
        self.conv2 = WS(nn.Conv2d(hidn_ch, out_ch, kernel_size=3,
                           stride=1, padding=1))

        self.scale = scale
        self.upsample = F.interpolate #learnt upsample
        self.noise_factor = 0.01
        
        self.adaIN1 = self.adaIN()
        self.adaIN2 = self.adaIN()
 
    def forward(self, w, x):
        
        if self.first == False:
            x = nn.ReLU(self.conv1(x))
            
            x = x + nn.Parameter(torch.zeros(1, self.out_ch, 1, 1)) + \
                torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]))
            x = self.adaIN1(x,w)
            
            x = nn.ReLU(self.conv2(x))
            x = x + nn.Parameter(torch.zeros(1, self.out_ch, 1, 1)) + \
                torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]))
            x = self.adaIN2(x,w)
        
        else:
            x = x + nn.Parameter(torch.zeros(1, self.out_ch, 1, 1)) + \
                torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]))
            x = self.adaIN1(x,w)
            x = nn.ReLU(self.conv1(x))
            x = x + nn.Parameter(torch.zeros(1, self.out_ch, 1, 1)) + \
                torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]))
           
            
        return x

        
        
class Generator(nn.Module):
    
    def __init__(self, z_dim, w_dim, in_ch, img_ch = 3):
        super().__init__()
        
        self.map = z_to_w(z_dim, w_dim)
        self.rgb = WS(nn.Conv2d(in_ch, img_ch, kernel_size = 1, stride = 1, 
                                padding = 0))
        
        self.prog_blocks = nn.ModuleList([SynthLayer(in_ch, in_ch, in_ch, 
                                                     first = True)])
        
        self.rgb_layers = nn.ModuleList([self.rgb])
        
        self.init_constant = nn.Parameter(torch.ones(1, in_ch, 4,4))
        
        
        #initiate an empty untrained list of blocks
        for i in range(len(factors)-1):
            conv_in_c = int(in_ch * factors[i])
            conv_out_c = int(in_ch * factors[i+1])
            
            self.prog_blocks.append(SynthLayer(conv_in_c, conv_out_c, 
                                               conv_out_c))
            self.rgb_layers.append(WS(nn.Conv2d(conv_in_c, img_ch, 
                                                kernel_size = 1, stride = 1, 
                                                padding = 0)))
        
    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1-alpha ) * upscaled)
    
    def forward(self, noise, alpha, steps):
        w = self.map(noise)
        up = self.prog_blocks[0](self.init_constant, w)
        x = adaIN()(up, w)
        if steps == 0:
            up = self.rgb_layers[steps](up)
            return up
        
        for step in range(steps):
            up = F.interpolate(x, scale_factor =2, mode = 'bilinear')
            x = self.prog_blocks[step](up, w)
            
        up_final = self.rgb_layers[steps-1](up)
        out_final = self.rgb_layers[steps](x)
        
        return self.fade_in(alpha, up_final, out_final)
    


class DiscLayer(nn.Module):
    def __init__(self, in_ch, out_ch, hidn_ch, first = False):
        super(DiscLayer, self).__init__()
        self.first = first
        if self.first == True:
            self.conv1 = WS(nn.Conv2d(in_ch + 1, hidn_ch, kernel_size = 3,
                                      padding = 1))
            self.conv2 = WS(nn.Conv2d(hidn_ch, hidn_ch, kernel_size = 4,
                                      padding = 0, stride = 1))
            
            self.conv3 = WS(nn.Conv2d(hidn_ch, out_ch, kernel_size = 1, 
                                      padding = 0, stride = 1))
            
        else:
            
            self.conv1 = WS(nn.Conv2d(in_ch, hidn_ch))
            self.conv2 = WS(nn.Conv2d(hidn_ch, out_ch))
            self.conv3 = None
        
        
    def forward(self, x):
        
        if self.first == True:
            x = nn.ReLU(self.conv1(x))
            x = nn.ReLU(self.conv2(x))
            x = nn.ReLU(self.conv3(x))
        else:
            x = nn.ReLU(self.conv1(x))
            x = nn.ReLU(self.conv2(x))
        
        return x
        
class Discriminator(nn.Module):
    
    def __init__(self, in_ch, img_ch = 3):
        super(Discriminator, self).__init__()
        
        self.prog_blocks = nn.ModuleList([])
        
        self.rgb_layers = nn.ModuleList([])
        
        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(in_ch * factors[i])
            conv_out = int(in_ch * factors[i-1])
            
            self.prog_blocks.append(DiscLayer(conv_in, conv_out, conv_out))
            self.rgb_layers.append(WS(nn.Conv2d(img_ch, in_ch, kernel_size=1, 
                                                stride = 1, padding = 0)))
            
        self.rgb_layers.append(WS(nn.Conv2d(img_ch, in_ch, kernel_size = 1,
                                      stride = 1, padding = 0)))
        self.downsample = nn.AvgPool2d(kernel_size=2, stride = 2)
        
        self.final = DiscLayer(conv_in, conv_in, conv_in, True)
        
    def fade_in(self, alpha, downscaled, out):
        
        return alpha * out + (1 - alpha) * downscaled
    
    def minbatch_std(self, x):
        batch_stat = (torch.std(x, dim=0).mean().repeat(x.shape[0],x, 
                                                        x.shape[2], x.shape[3]))
        
        return torch.cat([x, batch_stat], dim=1)
    
    def forward(self, x, alpha, steps):
        
        down = self.prog_blocks[0](x)
        
        current_step = len(self.prog_blocks) - steps
        
        
        if steps == 0:
            down = self.rgb_layers[0](down)
            return down
        
        dscaled = \
            self.leaky(self.rgb_layers[current_step+1](self.downsample(x)))
        down = self.fade_in(alpha, dscaled, down)
        for step in range(current_step + 1, len(self.prog_blocks)):
            down = self.prog_blocks[step](down)
            
            down = self.downsample(down)
            
        down = self.minbatch_std(down)
        
        return self.final_block(down).view(down.shape[0], -1)
    
    def loss_function(disc, real, fake, alpha, step, device = "cuda"):
        
        BATCH_SIZE, C, H, W = real.shape
        
        beta = torch.rand((BATCH_SIZE, 1,1, 1)).repeat(1, C, H, W).to(device)
        interpolated_img = real * beta + fake.detach()* (1-beta)
        interpolated_img.requires_grad_(True)
        
        mixed_scores = disc(interpolated_img, alpha, step)
        
        gradient = torch.autograd.grad(inputs = interpolated_img, outputs
                                       = mixed_scores, 
                                       grad_outputs=torch.ones_like(mixed_scores),
                                       create_graph = True, retain_graph = True)[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        
        penalty = torch.mean((gradient_norm - 1)** 2)
        return penalty
    
    
    
    
        
        
        
            
            