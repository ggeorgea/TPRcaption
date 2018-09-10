import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
    
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512,num_layers=2):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embed_size, hidden_size,num_layers,batch_first = True,dropout = 0.4)
        self.drop = nn.Dropout()
        self.embed_captions = torch.nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        self.batch_size = features.shape[0]
        #captions are batch, length
        #features are batch, length
        embedded_captions = self.embed_captions(  torch.tensor( (captions.float()).view(captions.size()[0],captions.size()[1]), dtype=torch.long).cuda()    )
        input = torch.cat((features.view(features.size()[0],1,features.size()[1]),embedded_captions),dim = 1)
        lstm_out, self.hidden = self.lstm(input)#, self.hidden)
        lstm_out = self.drop(lstm_out)
        lstm_out = lstm_out[:,:-1,:]
        tag_outputs = self.fc(lstm_out)
        return tag_outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        input = inputs
        lstm_out, self.hidden = self.lstm(input)#, self.hidden)
        captionList = []
        lastword = self.embed_captions(torch.tensor((torch.zeros(1)), dtype=torch.long).cuda())
        for i in range (1,max_len):
            lastword = lastword.view(1,1,lastword.shape[len(lastword.shape)-1])
            lstm_out, self.hidden = self.lstm(lastword, self.hidden)
            tag_outputs = self.fc(lstm_out)
            indices = torch.argmax(tag_outputs,2)
            lastword = self.embed_captions(indices.type(torch.LongTensor).cuda())     
            if indices.data.tolist()[0][0] == 1 :
                return captionList
            else: 
                #appendtensor
                captionList.append( torch.argmax(tag_outputs[:,:,3:],2).data.tolist()[0][0] +3)
        return captionList