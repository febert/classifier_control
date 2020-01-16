
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from classifier_control.classifier.utils.layers import ConvBlockEnc, ConvBlockDec


def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if isinstance(m, nn.Conv2d):
        pass    # by default PyTorch uses Kaiming_Normal initializer


class GetIntermediatesSequential(nn.Sequential):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, input):
        """Computes forward pass through the network outputting all intermediate activations with final output."""
        skips = []
        for i, module in enumerate(self._modules.values()):
            input = module(input)

            if i % self.stride == 0:
                skips.append(input)
            else:
                skips.append(None)
        return input, skips[:-1]

class ConvEncoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self._hp = hp

        self.n = hp.builder.get_num_layers(hp.img_sz) - 1
        self.net = GetIntermediatesSequential(hp.skips_stride) if hp.use_skips else nn.Sequential()
        
        if hp.goal_cond:
          input_c = hp.input_nc * 2
        else:
          input_c = hp.input_nc 
        

        print('l-1: indim {} outdim {}'.format(input_c, hp.ngf))
        self.net.add_module('input', ConvBlockEnc(in_dim=input_c, out_dim=hp.ngf, normalization=None,
                                                  builder=hp.builder))
        for i in range(self.n - 3):
            filters_in = hp.ngf * 2 ** i
            self.net.add_module('pyramid-{}'.format(i),
                                ConvBlockEnc(in_dim=filters_in, out_dim=filters_in * 2, normalize=hp.builder.normalize,
                                             builder=hp.builder))
            print('l{}: indim {} outdim {}'.format(i, filters_in, filters_in*2))

        # add output layer
        self.net.add_module('head', nn.Conv2d(hp.ngf * 2 ** (self.n - 3), hp.nz_enc, 4))
        print('l out: indim {} outdim {}'.format(hp.ngf * 2 ** (self.n - 3), hp.nz_enc))

        self.net.apply(init_weights_xavier)

    def get_output_size(self):
        # return (self._hp.nz_enc, self._hp.img_sz[0]//(2**self.n), self._hp.img_sz[1]//(2**self.n))
        return (64, 9, 13)   # todo calc this, fix the padding in the convs!

    def forward(self, input):
        return self.net(input)
      
class ConvDecoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self._hp = hp

        self.n = hp.builder.get_num_layers(hp.img_sz) - 1
        self.net = GetIntermediatesSequential(hp.skips_stride) if hp.use_skips else nn.Sequential()

#         print('l-1: indim {} outdim {}'.format(64, hp./))
        self.net.add_module('head', nn.ConvTranspose2d(64, 32, 4))
        
        
        for i in range(self.n - 3):
            filters_in = 32 // 2 ** i
            self.net.add_module('pyramid-{}'.format(i),
                                ConvBlockDec(in_dim=filters_in, out_dim=filters_in // 2, normalize=hp.builder.normalize,
                                             builder=hp.builder))
            print('l{}: indim {} outdim {}'.format(i, filters_in, filters_in // 2))
            
            
        self.net.add_module('input', ConvBlockDec(in_dim=8, out_dim=hp.input_nc, normalization=None,
                                                  builder=hp.builder))

        # add output layer
        
#         print('l out: indim {} outdim {}'.format(hp.ngf * 2 ** (self.n - 3), hp.nz_enc))

        self.net.apply(init_weights_xavier)

    def get_output_size(self):
        # return (self._hp.nz_enc, self._hp.img_sz[0]//(2**self.n), self._hp.img_sz[1]//(2**self.n))
        return (3, 64, 64)   # todo calc this, fix the padding in the convs!

    def forward(self, input):
        return self.net(input)