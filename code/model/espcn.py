from model import common

import torch.nn as nn

def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return ESPCN(args, dilated.dilated_conv)
    else:
        return ESPCN(args)

class ESPCN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(ESPCN_RELU, self).__init__()

        n_layers = len(args.kernel_size_list)
        n_feats_hidden = args.n_feats_list[:]
        kernel_size = args.kernel_size_list[:]

        scale = args.scale[0]
        act = nn.Tanh if args.actLast == 'tanh' else nn.ReLU(True) 

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        m_head = [conv(args.n_colors, n_feats[0], kernel_size[0]), nn.ReLU(True)]
  
        # define body module
        m_body = []
        for i in range(1,n_layers-1) :
            if i < n_layers-2:
                m_body.extend([conv(n_feats[i-1], n_feats[i], kernel_size[i]), nn.ReLU(True)]
            else:
                m_body.extend([conv(n_feats[i-1], n_feats[i], kernel_size[i]), act]
       
        # define tail module
        m_tail = [conv(n_feats[-1], scale * scale * args.n_colors, 3, bias)), nn.PixelShuffle(scale)) ]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x = self.add_mean(x)
        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

