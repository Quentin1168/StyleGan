# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 08:51:17 2023

@author: blobf
"""

import torch
import torch.nn as nn
import torch.utils as utils
import torchvision
import numpy as np
import os
import modules
import dataset
import tqdm

import dataset

EPOCHS = [50]*6


class Train():
    
    def __init__(self, lr, epochs, train_path, step, alpha, gen_opt, disc_opt,
                 batches):
        self.batches = batches
        self.epochs = epochs
        
        self.train_path = train_path
            
        self.gen_opt = gen_opt
        
        self.disc_opt = disc_opt
        
        

    def training_func(self, disc, gen, data, step, alpha):
        
        epoch = 0
        
        while epoch != self.epochs:
                 
            for i, _ in data:
                i = i.view(-1, 3, 256, 256)
                batch_sz = i.shape[0]
                
                i = i.to("cuda")
                noise = torch.randn(32, 100).to("cuda")
                fake = gen(noise, alpha, step)
                disc_real = disc(i, alpha, step)
                disc_fake = disc(fake.detach(), alpha, step)
                gp = modules.loss_function(disc, i, fake, alpha, step, "cuda")
                
                loss = (-(torch.mean(disc_real) - torch.mean(disc_fake))+ 10 * gp + (0.001) * torch.mean(disc_real ** 2))
                
                disc.zero_grad()
                loss.backward()
                self.disc_opt.step()
                
                gen_fake = disc(fake, alpha, step)
                
                loss_gen = -torch.mean(gen_fake)
                
                gen.zero_grad()
                loss_gen.backward()
                self.gen_opt.step()
                
                alpha = batch_sz / (50 * 0.5 * len(self.training_data))
                
                alpha = min(alpha, 1)
                
        return alpha
    
    def train(self, gen, disc):
        gen.train()
        disc.train()
        
        step = 1
        
        for epochs in EPOCHS:
            alpha = 1e-7
            data = dataset.DataLoader(self.train_path, 4 * 2 ** step)

            training_data = \
                utils.data.DataLoader(data, batch_size = self.batches[0], shuffle = True)
                
            for epoch in range(epochs):
                print("Epoch " + epoch + " of " + epochs)
                alpha = self.training_func(disc, gen, training_data, step, alpha)
            step += 1                    
                    
                
                        