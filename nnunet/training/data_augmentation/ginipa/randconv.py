import torch
import torch.nn as nn 
from .gin import GradlessGCReplayNonlinBlock


class RandConv(nn.Module):
    def __init__(self, out_channel = 3,
                 in_channel = 3,
                 interm_channel = 2,
                 scale_pool = [1, 3 ],
                 n_layer = 4,
                 out_norm = 'frob', 
                 **kwargs):
        '''
        GIN
        '''
        super(RandConv, self).__init__()
        self.scale_pool = scale_pool # don't make it tool large as we have multiple layers
        self.n_layer = n_layer
        self.layers = []
        self.out_norm = out_norm
        self.out_channel = out_channel

        self.layers = GradlessGCReplayNonlinBlock(out_channel = interm_channel, 
                                                  in_channel = in_channel,
                                                  scale_pool = scale_pool, 
                                                  use_act=False, 
                                                  layer_id = 0).cuda()


    def forward(self, x_in):
        # random augmentation at 50%
        nb, nc, nx, ny = x_in.shape
        alphas = torch.rand(nb)[:, None, None, None] # nb, 1, 1, 1
        alphas = alphas.repeat(1, nc, 1, 1).cuda() # nb, nc, 1, 1
        x = self.layer(x_in)
        mixed = alphas * x + (1.0 - alphas) * x_in

        # energy-preserving, I guess? 
        if self.out_norm == 'frob':
            _in_frob = torch.norm(x_in.contiguous().view(nb, nc, -1), dim = (-1, -2), p = 'fro', keepdim = False)
            _in_frob = _in_frob[:, None, None, None].repeat(1, nc, 1, 1)
            _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim = (-1,-2), p = 'fro', keepdim = False)
            _self_frob = _self_frob[:, None, None, None].repeat(1, self.out_channel, 1, 1)
            mixed = mixed * (1.0 / (_self_frob + 1e-5 ) ) * _in_frob
        return mixed

