import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
#make embedding full vocab instead
#don't fit to my caps - to theirs!! 
#dropout
#other OPT

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
    def __init__(self, vocab_size, embed_size=256, dSize=25, hidden_size=512,num_layers=1):
        #their d is 25
        #they dont embed words??? i can surely
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        #self.hidden_size = hidden_size
        self.dSize = dSize

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.lstmS = nn.LSTM(embed_size+dSize*dSize+dSize, dSize*dSize,num_layers,batch_first = True,dropout = 0.4)
        self.lstmU = nn.LSTM(embed_size+dSize*dSize+dSize, dSize,num_layers,batch_first = True,dropout = 0.4)
        self.drop = nn.Dropout()
        self.embed_captions = torch.nn.Embedding(vocab_size, embed_size)
        #TODO hidden_Size alert return to these later!!!
        self.fc = nn.Linear(self.dSize*self.dSize, vocab_size)
        self.SzeroLinear = nn.Linear(embed_size,self.dSize*self.dSize)
        self.wu = nn.Linear(self.dSize,self.dSize*self.dSize)

    def forward(self, features, captions):
        #TODO ADD DSTARTWORD PROPERLY
        self.batch_size = features.shape[0]
        #captions are batch, length
        #features are batch, length


        #OPT should jusat be a linear layer, bo embveding
        embedded_captions = self.embed_captions(  torch.tensor( (captions.float()).view(captions.size()[0],captions.size()[1]), dtype=torch.long).cuda()    )
        inputS = features

        lastword = self.embed_captions(torch.tensor((torch.zeros(1)), dtype=torch.long).cuda())
        lastword = lastword.view(1,1,lastword.shape[len(lastword.shape)-1])

        wordstart = lastword.repeat(self.batch_size,1,1)#1,x,1
        
        # #test
        # print(embedded_captions)
        # print(wordstart)
        # print(embedded_captions[:,0:1,:])
        # print(embedded_captions.shape)
        # print(wordstart.shape)
        # print(embedded_captions[:,0:1,:].shape)
        # return

        p0zeroes = torch.zeros(self.batch_size,1,self.dSize).cuda()#1,x,x
        sStartHidden = (self.SzeroLinear(features)).view(self.num_layers,self.batch_size,self.dSize*self.dSize)#

        # fSIn = torch.cat((p0zeroes, wordstart),dim = 2)
        # fUin = torch.cat((sStartHidden.view(self.batch_size ,self.num_layers,self.dSize*self.dSize), wordstart),dim = 2)
        fIn = torch.cat((p0zeroes, wordstart,sStartHidden.view(self.batch_size ,self.num_layers,self.dSize*self.dSize)),dim = 2)
        #CUDA EXEC FAILED ERRORS CAN BE NO .CUDA
        lstSOut,self.hidden = self.lstmS(fIn,( torch.zeros(sStartHidden.shape).cuda(),sStartHidden))
        lstUOut,self.hiddenU = self.lstmU(fIn)

        # lstmS_out = []
        # lstmS_out.append(lstSOut)
        # lstmU_out = []
        # lstmU_out.append(lstUOut)
        wordOut = []
        wordOut.append(torch.zeros(self.batch_size,1,self.dSize*self.dSize).cuda())

        #go here actually!
        matrix2 = lstSOut.view(self.batch_size,self.dSize,self.dSize)
        matrix1 = torch.eye(matrix2.shape[1],matrix2.shape[2]).view(1,matrix2.shape[1],matrix2.shape[2]).repeat(matrix2.shape[0],1,1).cuda()
        #blockDiagonal = torch.bmm(matrix1.view(matrix1.shape[0],-1,1),matrix2.view(matrix2.shape[0],1,-1)).view(matrix1.shape[0],*(matrix1[0].size()+matrix2[0].size())).permute([0,1,3,2,4]).view(matrix1.size(0),matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))
        blockDiagonal = torch.bmm(matrix1.view(matrix1.shape[0],-1,1),matrix2.view(matrix2.shape[0],1,-1)).view(matrix1.shape[0],*(matrix1[0].size()+matrix2[0].size())).permute([0,1,3,2,4]).reshape(matrix1.size(0),matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))

        ut = self.wu(lstUOut).view(self.batch_size,1,self.dSize*self.dSize)
        ft = torch.bmm(ut,blockDiagonal).view(self.batch_size,1,self.dSize*self.dSize)
        indices = torch.argmax(self.fc(ft),2)    
        xprev = self.embed_captions(indices.type(torch.LongTensor).cuda())  
        #aaahhhh
        wordOut.append(ft)

        for inty in range(1,captions.size()[1]-1):
            
            # fSIn = torch.cat((p0zeroes, wordstart),dim = 2)
            # fUin = torch.cat((sStartHidden.view(self.batch_size ,self.num_layers,self.dSize*self.dSize), wordstart),dim = 2)
            #fInpt = torch.cat((p0zeroes, wordstart,sStartHidden.view(self.batch_size ,self.num_layers,self.dSize*self.dSize)),dim = 2)
            
            fInpt = torch.cat((self.hiddenU[1].view(self.batch_size,1,-1),embedded_captions[:,inty:inty+1,:], self.hidden[1].view(self.batch_size,1,-1)), dim = 2)
            #IF YOU WANT TO LEARN WITH YOUR GENERATED CAPS AS  OPPOSED TO REVERTING
            #fInpt = torch.cat((self.hiddenU[1].view(self.batch_size,1,-1),xprev, self.hidden[1].view(self.batch_size,1,-1)), dim = 2)

            # sInStep = torch.cat((self.hiddenU[1].view(self.batch_size,1,-1),xprev), dim = 2)
            # uInStep = torch.cat((self.hidden[1].view(self.batch_size,1,-1),xprev),dim = 2)
            lstSOut,self.hidden = self.lstmS(fInpt,self.hidden)
            lstUOut,self.hiddenU = self.lstmU(fInpt,self.hiddenU)
          


            matrix2 = lstSOut.view(self.batch_size,self.dSize,self.dSize)
            #matrix1 = torch.eye(matrix2.shape[1],matrix2.shape[2]).view(1,matrix2.shape[1],matrix2.shape[2]).repeat(matrix2.shape[0],1,1).cuda()
            #blockDiagonal = torch.bmm(matrix1.view(matrix1.shape[0],-1,1),matrix2.view(matrix2.shape[0],1,-1)).view(matrix1.shape[0],*(matrix1[0].size()+matrix2[0].size())).permute([0,1,3,2,4]).view(matrix1.size(0),matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))
            blockDiagonal = torch.bmm(matrix1.view(matrix1.shape[0],-1,1),matrix2.view(matrix2.shape[0],1,-1)).view(matrix1.shape[0],*(matrix1[0].size()+matrix2[0].size())).permute([0,1,3,2,4]).reshape(matrix1.size(0),matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))

            ut = self.wu(lstUOut).view(self.batch_size,1,self.dSize*self.dSize)
            ft = torch.bmm(ut,blockDiagonal).view(self.batch_size,1,self.dSize*self.dSize)
            indices = torch.argmax(self.fc(ft),2)   



            # multOut = lstSOut #

            # indices = torch.argmax(self.fc(multOut),2)
            xprev = self.embed_captions(indices.type(torch.LongTensor).cuda()) 
            wordOut.append(ft)
        #     lstmS_out.append(lstSOut)
        #     lstmU_out.append(lstUOut)

        # lstmS_out = torch.stack(lstmS_out)
        wordOut = torch.stack(wordOut)
        # lstmS_out = lstmS_out.view(inputS.size()[0],inputS.size()[1],self.hidden_size)
        # lstmU_out = torch.stack(lstmU_out)
        # lstmU_out = lstmU_out.view(inputS.size()[0],inputS.size()[1],self.hidden_size)
        #======================================================
        #####!!!!lstmS_out = self.drop(lstmS_out)
        ####lstmS_out = lstmS_out[:,:-1,:]
        tag_outputs = self.fc(wordOut)
        return tag_outputs

    def sample(self, inputs, states=None, max_len=20):
        #" accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
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



	# def sample2(self, inputs, captions,max_len=20):
	# 	#TODO ADD DSTARTWORD PROPERLY
	# 	self.batch_size = features.shape[0]
	# 	#captions are batch, length
	# 	#features are batch, length
	# 	embedded_captions = self.embed_captions(  torch.tensor( (captions.float()).view(captions.size()[0],captions.size()[1]), dtype=torch.long).cuda()    )
	# 	inputS = features
	# 	lastword = self.embed_captions(torch.tensor((torch.zeros(1)), dtype=torch.long).cuda())
	# 	lastword = lastword.view(1,1,lastword.shape[len(lastword.shape)-1])
	# 	wordstart = lastword.repeat(self.batch_size,1,1)#1,x,1
	# 	p0zeroes = torch.zeros(self.batch_size,1,self.dSize).cuda()#1,x,x
	# 	sStartHidden = (self.SzeroLinear(features)).view(self.num_layers,self.batch_size,self.dSize*self.dSize)#
	# 	fIn = torch.cat((p0zeroes, wordstart,sStartHidden.view(self.batch_size ,self.num_layers,self.dSize*self.dSize)),dim = 2)
	# 	#CUDA EXEC FAILED ERRORS CAN BE NO .CUDA
	# 	lstSOut,self.hidden = self.lstmS(fIn,( torch.zeros(sStartHidden.shape).cuda(),sStartHidden))
	# 	lstUOut,self.hiddenU = self.lstmU(fIn)
	# 	wordOut = []
	# 	wordOut.append(torch.zeros(self.batch_size,1,self.dSize*self.dSize).cuda())
	# 	matrix2 = lstSOut.view(self.batch_size,self.dSize,self.dSize)
	# 	matrix1 = torch.eye(matrix2.shape[1],matrix2.shape[2]).view(1,matrix2.shape[1],matrix2.shape[2]).repeat(matrix2.shape[0],1,1).cuda()
	# 	blockDiagonal = torch.bmm(matrix1.view(matrix1.shape[0],-1,1),matrix2.view(matrix2.shape[0],1,-1)).view(matrix1.shape[0],*(matrix1[0].size()+matrix2[0].size())).permute([0,1,3,2,4]).reshape(matrix1.size(0),matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))
	# 	ut = self.wu(lstUOut).view(self.batch_size,1,self.dSize*self.dSize)
	# 	ft = torch.bmm(ut,blockDiagonal).view(self.batch_size,1,self.dSize*self.dSize)
	# 	indices = torch.argmax(self.fc(ft),2)    
	# 	xprev = self.embed_captions(indices.type(torch.LongTensor).cuda())  
	# 	wordOut.append(ft)
	# 	for inty in range(1,captions.size()[1]-1):	       
	# 	    fInpt = torch.cat((self.hiddenU[1].view(self.batch_size,1,-1),xprev, self.hidden[1].view(self.batch_size,1,-1)), dim = 2)
	# 		lstSOut,self.hidden = self.lstmS(fInpt,self.hidden)
	# 	    lstUOut,self.hiddenU = self.lstmU(fInpt,self.hiddenU)	     
	# 	    matrix2 = lstSOut.view(self.batch_size,self.dSize,self.dSize)
	# 	    blockDiagonal = torch.bmm(matrix1.view(matrix1.shape[0],-1,1),matrix2.view(matrix2.shape[0],1,-1)).view(matrix1.shape[0],*(matrix1[0].size()+matrix2[0].size())).permute([0,1,3,2,4]).reshape(matrix1.size(0),matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))
	# 	    ut = self.wu(lstUOut).view(self.batch_size,1,self.dSize*self.dSize)
	# 	    ft = torch.bmm(ut,blockDiagonal).view(self.batch_size,1,self.dSize*self.dSize)
	# 	    indices = torch.argmax(self.fc(ft),2)   
	# 	    xprev = self.embed_captions(indices.type(torch.LongTensor).cuda()) 
	# 	    wordOut.append(ft)
	# 	wordOut = torch.stack(wordOut)
	# 	tag_outputs = self.fc(wordOut)
	# 	return tag_outputs