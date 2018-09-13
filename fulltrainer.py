import os
import sys
import math
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from pycocotools.coco import COCO
from COCOClass import CoCoDataset
from torchvision import transforms
from TPRmodel import EncoderCNN, DecoderRNN
def train():
    vocab_from_file = True    
    num_epochs = 2        
    transform_train = transforms.Compose([ transforms.Resize(256), transforms.RandomCrop(224),                      
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    batch_size = 50     #100         # 300
    vocab_threshold = 5       # 5
    img_folder = './cocoapi/images/train2014/'
    annotations_file = './cocoapi/annotations/captions_train2014.json'
    mode='train'
    vocab_file='./vocab.pkl'
    start_word="<start>"
    end_word="<end>"
    unk_word="<unk>"
    dataset = CoCoDataset(transform=transform_train, mode='train', batch_size=batch_size, vocab_threshold=vocab_threshold,
        vocab_file=vocab_file, start_word=start_word, end_word=end_word, unk_word=unk_word, annotations_file=annotations_file,
        vocab_from_file=vocab_from_file, img_folder=img_folder)
    indices = dataset.get_train_indices()
    initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
    data_loader = data.DataLoader(dataset=dataset, num_workers=0, 
        batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
        batch_size=dataset.batch_size,
        drop_last=False))
    vocab_size = len(data_loader.dataset.vocab)
    encoder = EncoderCNN()
    decoder = DecoderRNN(vocab_size)
    # encoder_file = "locencoder-1.pkl" 
    # decoder_file = "locdecoder-1.pkl"
    # encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file)))
    # decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.embed.parameters())
    optimizer =     opt = torch.optim.Adam(params = params,lr=0.001)
    total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)
    for epoch in range(1, num_epochs+1):
        for i_step in range(1, total_step+1):        
            indices = data_loader.dataset.get_train_indices()
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            data_loader.batch_sampler.sampler = new_sampler

            images, captions = next(iter(data_loader))
            images = images.to(device)
            captions = captions.to(device)

            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(images)
            outputs = decoder(features, captions)
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            loss.backward()
            optimizer.step()
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch, num_epochs, i_step, total_step, loss.item(), np.exp(loss.item()))
            print('\r' + stats, end="")
            sys.stdout.flush()
            print('\r' + stats)
            if i_step % 10 == 0:
                torch.save(decoder.state_dict(), os.path.join('./models', 'ttdecoder-%d.pkl' % (epoch+0)))
                torch.save(encoder.state_dict(), os.path.join('./models', 'ttencoder-%d.pkl' % (epoch+0)))
        if epoch % 1 == 0:
            torch.save(decoder.state_dict(), os.path.join('./models', 'ttdecoder-%d.pkl' % (epoch+0)))
            torch.save(encoder.state_dict(), os.path.join('./models', 'ttencoder-%d.pkl' % (epoch+0)))

train()

