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
    def __init__(self, vocab_size, embed_size=256, hidden_size=512,num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.lstmS = nn.LSTM(embed_size+hidden_size, hidden_size,num_layers,batch_first = True,dropout = 0.4)
        self.lstmU = nn.LSTM(embed_size+hidden_size, hidden_size,num_layers,batch_first = True,dropout = 0.4)
        self.drop = nn.Dropout()
        self.embed_captions = torch.nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.SzeroLinear = nn.Linear(embed_size,hidden_size)
    
    def forward(self, features, captions):
        self.batch_size = features.shape[0]
        #captions are batch, length
        #features are batch, length
        #\/\/\/\/!!!!
        embedded_captions = self.embed_captions(  torch.tensor( (captions.float()).view(captions.size()[0],captions.size()[1]), dtype=torch.long).cuda()    )
        # inputS = torch.cat((features.view(features.size()[0],1,features.size()[1]),embedded_captions),dim = 1)
        inputS = features
        #.view(1,features.size()[0],features.size()[1])

        #lstmS_out, self.hidden = self.lstmS(inputS)#, self.hidden)
        #=====================================================
        #start word
        lastword = self.embed_captions(torch.tensor((torch.zeros(1)), dtype=torch.long).cuda())
        
        lastword = lastword.view(1,1,lastword.shape[len(lastword.shape)-1])


        wordstart = lastword.repeat(self.batch_size,1,1)#1,x,1
        p0zeroes = torch.zeros(self.batch_size,1,self.hidden_size).cuda()#1,x,x
        fSIn = torch.cat((p0zeroes, wordstart),dim = 2)
        sStartHidden = (self.SzeroLinear(features)).view(self.num_layers,self.batch_size,self.hidden_size)#
        fUin = torch.cat((p0zeroes, wordstart),dim = 2)
        print(sStartHidden.shape)
        lstSOut,self.hidden = self.lstmS(fSIn,sStartHidden)
        lstUOut,self.hiddenU = self.lstmU(fUin)

        #temp
        lstmS_out.append(wordstart)

        lstmS_out = []
        lstmU_out = []

        # #first attempt at first go over
        # lstUOut =  torch.zeros(1,self.batch_size,self.hidden_size).cuda()
        # lstmS_out = []
        # iStartS = inputS.view(-1,inputS.size()[0],inputS.size()[1])
        # # lstSOut =  torch.zeros(iStartS.size()[0],iStartS.size()[1],self.hidden_size).cuda()
        # lstSOut =  torch.zeros(iStartS.size()[0],iStartS.size()[1],self.hidden_size).cuda()
        # fSIn = torch.cat((iStartS,lstSOut ),dim = 2)
        # #is features, empty hidden state
        # #should be startword, empty hidden state
        # #should be lstm0(input = startchar, hidden = linear of features to hiddensize)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # #TODO TODO TODO TODO
        # lstSOut,self.hidden = self.lstmS(fSIn)
        # lstmS_out.append(lstSOut)
        # #p0 zero vector
        # lstmU_out = []
        # fUin = fSin # should change ;ater
        # lstUOut,self.hiddenU = self.lstmU(fUin)
        # lstmU_out.append(lstUOut)



        for inty in range(1,captions.size()[1]):
            #this should be the * of lstuout and this
            finalO  = lstSOut
            inp = finalO
            #inputS[inty]
           
            sIn = torch.cat((inp.view(-1,inp.size()[0],inp.size()[1]),lstUOut ),dim = 2)
            lstSOut,self.hidden = self.lstmS(sIn,self.hidden)
            lstmS_out.append(lstSOut)

            uIn = torch.cat((inp.view(-1,inp.size()[0],inp.size()[1]),lstSOut ),dim = 2)
            lstUOut,self.hiddenU = self.lstmU(uIn,self.hiddenU)
            lstmU_out.append(lstUOut)

        lstmS_out = torch.stack(lstmS_out)
        lstmS_out = lstmS_out.view(inputS.size()[0],inputS.size()[1],self.hidden_size)
        lstmU_out = torch.stack(lstmU_out)
        lstmU_out = lstmU_out.view(inputS.size()[0],inputS.size()[1],self.hidden_size)



        #======================================================
        lstmS_out = self.drop(lstmS_out)
        lstmS_out = lstmS_out[:,:-1,:]
        tag_outputs = self.fc(lstmS_out)


        return tag_outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        inputS = inputs
        

        #TODO change to reflect above
        lstmS_out, self.hidden = self.lstmS(inputS)#, self.hidden)




        captionList = []
        lastword = self.embed_captions(torch.tensor((torch.zeros(1)), dtype=torch.long).cuda())
        for i in range (1,max_len):
            lastword = lastword.view(1,1,lastword.shape[len(lastword.shape)-1])
            lstmS_out, self.hidden = self.lstmS(lastword, self.hidden)
            tag_outputs = self.fc(lstmS_out)
            indices = torch.argmax(tag_outputs,2)
            lastword = self.embed_captions(indices.type(torch.LongTensor).cuda())     
            if indices.data.tolist()[0][0] == 1 :
                return captionList
            else: 
                #appendtensor
                captionList.append( torch.argmax(tag_outputs[:,:,3:],2).data.tolist()[0][0] +3)
        return captionList