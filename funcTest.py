import torch
def kronecker(matrix1, matrix2):
	#opp to online ex
	thing = torch.ger(matrix1.view(-1), matrix2.view(-1))
	thing = thing.reshape(*(matrix1.size() + matrix2.size()))
	thing = thing.permute([0, 2, 1, 3])
	thing = thing.reshape(matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1))
	return thing
def batchkronecker(matrix1, matrix2):
	thing = torch.bmm(matrix1.view(matrix1.shape[0],-1,1),matrix2.view(matrix2.shape[0],1,-1))
	thing = thing.view(matrix1.shape[0],*(matrix1[0].size()+matrix2[0].size()))
	thing = thing.permute([0,1,3,2,4])
	thing = thing.reshape(matrix1.size(0),matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))
	return thing#torch.bmm(matrix1.view(matrix1.shape[0],-1,1),matrix2.view(matrix2.shape[0],1,-1)).view(matrix1.shape[0],*(matrix1[0].size()+matrix2[0].size())).permute([0,1,3,2,4]).reshape(matrix1.size(0),matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))

# print("batch")
# m2 = torch.randn(4,2,2)#.eye(2,2)#.view(1,2,-1)
# m1 = torch.eye(2,2).view(1,2,2).repeat(4,1,1)#.view(1,2,-1)
# print(m1)
# print(m2)
# stb = torch.bmm(m1.view(m1.shape[0],-1,1),m2.view(m1.shape[0],1,-1))
# print(stb)
# stb = stb.view(m1.shape[0],*(m1[0].size()+m2[0].size()))
# print(stb)
# stb = stb.permute([0,1,3,2,4])
# print(stb)
# stb = stb.reshape(m1.size(0),m1.size(1) * m2.size(1), m1.size(2) * m2.size(2))
# print(stb)

# #vs
# print("\nclassic")
# m3 = m1[0]
# m4 = m2[0]
# print(m3)
# print(m4)
# stp = torch.ger(m3.view(-1),m4.view(-1))
# print(stp)
# stp = stp.reshape(*(m3.size() + m4.size()))
# print(stp)
# stp  = stp.permute([0, 2, 1, 3])
# print(stp)
# stp = stp.reshape(m3.size(0) * m4.size(0), m3.size(1) * m4.size(1))
# print(stp)
# print("\n\n\n",kronecker(m3,m4))
# print(batchkronecker(m1,m2))

# #print("\n",kronecker(torch.eye(2,2),torch.randn(2,2)))
matrix3 = torch.randn(3,1,4)

matrix2 = torch.randn(3,2,2)
matrix1 = torch.eye(matrix2.shape[1],matrix2.shape[2]).view(1,matrix2.shape[1],matrix2.shape[2]).repeat(matrix2.shape[0],1,1)
blockDiagonal = torch.bmm(matrix1.view(matrix1.shape[0],-1,1),matrix2.view(matrix2.shape[0],1,-1)).view(matrix1.shape[0],*(matrix1[0].size()+matrix2[0].size())).permute([0,1,3,2,4]).reshape(matrix1.size(0),matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))
# print(matrix3.shape)
# print(blockDiagonal.shape)
print(blockDiagonal.shape)
print(blockDiagonal)
f = torch.bmm(matrix3,blockDiagonal)
print(f)
print(f.shape)
#print(blockDiagonal.view(matrix3.shape[0],matrix3.shape[2],matrix3.shape[2]))
#print(f.shape)