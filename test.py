import torch
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity
# loc=torch.randn(7000,786).cuda()
# loc1=torch.randn(10,786).cuda()
#
# loc = loc.cpu()
#
#
#
# locc = loc.numpy()
# loc1c = loc1.numpy()
#
# loc=loc.unsqueeze(0)
# loc1=loc1.unsqueeze(1)
#
# b = cosine_similarity(loc1c, locc)


# rel_loc = torch.mean(loc, 1)
# a = torch.cosine_similarity(loc1, loc, dim=-1)
a= torch.tensor([[1,1,1,1,1],[2,2,2,2,2]],dtype=float)
b= torch.tensor([[4,4,4,4,4],[5,5,5,5,5]],dtype=float)
# c= torch.tensor([[7,7,7,7,7],[8,8,8,8,8]],dtype=float)
c = a + b
# d = torch.stack((a,b,c))
# e = d.permute(1,0,2)
# f = e.permute(1,0,2)
# g = f[0]
# h = f[1]
# i = f[2]
print(1)