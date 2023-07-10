from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms
import torch.cuda


device = "cuda" if torch.cuda.is_available() else "cpu"



class DataLoader(Dataset):
    
    
    def __init__(self, path, img_size):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                        transforms.resize((img_size, img_size))])
        
        dataloader = torchvision.datasets.ImageFolder(root = path, 
                                                       transform = transform)
        
        self.data = dataloader
        
    def __len__(self):
        return len(self.data)
        
                

    #returns the data_loader object
    def __getitem__(self, idx):
        return self.data[idx]
        
