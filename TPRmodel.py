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
        #TODO ADD DSTARTWORD PROPERLY
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
        sStartHidden = (self.SzeroLinear(features)).view(self.num_layers,self.batch_size,self.hidden_size)#

        fSIn = torch.cat((p0zeroes, wordstart),dim = 2)
        fUin = torch.cat((sStartHidden.view(self.batch_size ,self.num_layers,self.hidden_size), wordstart),dim = 2)
        #CUDA EXEC FAILED ERRORS CAN BE NO .CUDA
        #cudnn fail here
        #print(fSIn.shape)
        lstSOut,self.hidden = self.lstmS(fSIn,( torch.zeros(sStartHidden.shape).cuda(),sStartHidden))
        lstUOut,self.hiddenU = self.lstmU(fUin)

        #temp

        # firstout =wordstart#self.fc(wordstart)


        lstmS_out = []

        #lstmS_out.append(wordstart)
        lstmS_out.append(lstSOut)

        lstmU_out = []
        lstmU_out.append(lstUOut)
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
        # print(lstSOut.shape)
        indices = torch.argmax(self.fc(lstSOut),2)
        xprev = self.embed_captions(indices.type(torch.LongTensor).cuda())  
        # print(indices.shape)
        # print(xprev.shape)
        # print(self.hiddenU[1].shape)
        #xprev = (self.embed_captions(torch.tensor(self.fc(lstSOut), dtype=torch.long).cuda()))
     
        # print(xprev.shape)
        # print(self.fc(lstSOut).shape)
        # print(lstSOut.shape)
        # example lastword.view(1,1,lastword.shape[len(lastword.shape)-1])
        # example lastword = self.embed_captions(torch.tensor((torch.zeros(1)), dtype=torch.long).cuda())

        for inty in range(1,captions.size()[1]):
            
            #this should be the * of lstuout and this
            # finalO  = lstSOut
            # inp = finalO
            #inputS[inty]
            #sIn = torch.cat((inp.view(-1,inp.size()[0],inp.size()[1]),lstUOut ),dim = 2)
            #uIn = torch.cat((inp.view(-1,inp.size()[0],inp.size()[1]),lstSOut ),dim = 2)
            #temp \/\/
            sInStep = torch.cat((self.hiddenU[1].view(self.batch_size,1,-1),xprev), dim = 2)
            uInStep = torch.cat((self.hidden[1].view(self.batch_size,1,-1),xprev),dim = 2)
            lstSOut,self.hidden = self.lstmS(sInStep,self.hidden)
            lstUOut,self.hiddenU = self.lstmU(uInStep,self.hiddenU)
            
            matrix2 = lstSOut
            matrix1 = torch.eye(matrix2.shape[1],matrix2.shape[2]).view(1,matrix2.shape[1],matrix2.shape[2]).repeat(matrix2.shape[0],1,1)
            blockDiagonal = torch.bmm(matrix1.view(matrix1.shape[0],-1,1),matrix2.view(matrix2.shape[0],1,-1)).view(matrix1.shape[0],*(matrix1[0].size()+matrix2[0].size())).permute([0,1,3,2,4]).reshape(matrix1.size(0),matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))
            f = torch.bmm(lstUOut,f)
            print(f.shape)
            return 
            multOut = lstSOut #  

#            import torch
# def kronecker(matrix1, matrix2):
#     return torch.ger(matrix1.view(-1), matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute([0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1))
# m1 = torch.eye(2,2)
# m2 = torch.randn(2,2)
# print(m1)
# print(m2)
# print(kronecker(m1,m2))
            

            #TODO CURRENT IS MAKE THIS MATRIC MULTIPLY WORK PROPA
            #torch.bmm(lstSOut.view(self.batch_size),lstUOut.view(self.batch_size)) 
            
            print(multOut.shape,"!")
            indices = torch.argmax(self.fc(multOut),2)
            xprev = self.embed_captions(indices.type(torch.LongTensor).cuda()) 

            #xprev = (self.embed_captions(torch.tensor(self.fc(lstSOut), dtype=torch.long).cuda())).view(self.batch_size,1,-1)



            # xprev = self.embed_captions(self.fc(lstSOut))
            lstmS_out.append(lstSOut)
            lstmU_out.append(lstUOut)




        lstmS_out = torch.stack(lstmS_out)
        # lstmS_out = lstmS_out.view(inputS.size()[0],inputS.size()[1],self.hidden_size)
        # lstmU_out = torch.stack(lstmU_out)
        # lstmU_out = lstmU_out.view(inputS.size()[0],inputS.size()[1],self.hidden_size)

        #======================================================
        #####!!!!lstmS_out = self.drop(lstmS_out)
        ####lstmS_out = lstmS_out[:,:-1,:]
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