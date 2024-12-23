import torch
from torch import nn
from torch.nn import functional as F



class GeM(nn.Module):
    '''
    Code modified from the 2d code in
    https://amaarora.github.io/2020/08/30/gempool.html
    '''
    def __init__(self, kernel_size=8, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool1d(x.clamp(min=eps).pow(p), self.kernel_size).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'

class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_dim, emb_dim, out_dim):
        super(MultiLayerPerceptron, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, out_dim),
            nn.LayerNorm([out_dim]),
            nn.Dropout(0.25),
        )
    def forward(self, x):
        x = self.layers(x)
        return x
        

