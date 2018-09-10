import os
import sys
import math
import time
import torch
import requests
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from pycocotools.coco import COCO
from COCOClass import CoCoDataset
from model import EncoderCNN, DecoderRNN

batch_size = 1         
vocab_threshold = 5      
vocab_from_file = True   
transform = transforms.Compose([ transforms.Resize(256), transforms.RandomCrop(224),                      
    transforms.RandomHorizontalFlip(), transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
mode='val',
img_folder = './cocoapi/images/val2014/'
annotations_file = './cocoapi/annotations/captions_val2014.json'
vocab_file='./vocab.pkl'
start_word="<start>"
end_word="<end>"
unk_word="<unk>"

dataset = CoCoDataset(transform=transform, mode='val',batch_size=batch_size, vocab_threshold=vocab_threshold, vocab_file=vocab_file,
    start_word=start_word, end_word=end_word, unk_word=unk_word, annotations_file=annotations_file, 
    vocab_from_file=vocab_from_file, img_folder=img_folder)

indices = dataset.get_train_indices()

initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)

data_loader = data.DataLoader(dataset,num_workers=0,batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
    batch_size=dataset.batch_size,drop_last=False))

vocab_size = len(data_loader.dataset.vocab)
encoder = EncoderCNN()
decoder = DecoderRNN(vocab_size)

encoder_file = "ttencoder-1.pkl" 
decoder_file = "ttdecoder-1.pkl"
encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file)))
decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
total_loss=0
total_perp=0
sampl = 1000
for i_step in range(1, sampl): 
    indices = data_loader.dataset.get_train_indices()
    new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
    data_loader.batch_sampler.sampler = new_sampler
    images, captions = next(iter(data_loader))
    images = images.to(device)
    captions = captions.to(device)
    features = encoder(images)
    outputs = decoder(features, captions)
    loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
    total_loss += loss.item()
    total_perp += np.exp(loss.item())
print("\n\naverage loss:  %.4f"% (total_loss/sampl))
print("\naverage Perplexity: %5.4f\n" % (total_perp/sampl))