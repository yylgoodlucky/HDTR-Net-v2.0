import argparse
import numpy as np
import torch
from copy import deepcopy

from model import Teeth_Enhance
from utils.train_util import parse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, 
                    default='./exp/2024-09-10_18:28:38_teeth_enhance_resize128/ckpt/teeth_enhance_latest_net_g.pth')
parser.add_argument('--cfg', type=str, default='./opt/teeth_enhance.yml')

args = parser.parse_args()
print(f'Loaded params from {args.cfg}')
opt = parse(args.cfg)

def load_network(net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
            load_net = load_net[param_key]
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        net.load_state_dict(load_net, strict=strict)

def main():
    # init model 
    net_G = Teeth_Enhance(opt['network_g'])
    load_network(net_G, args.checkpoint)
    
    # input
    input = torch.randn(1,3,128,128)
    output = np.random.randn(1,3,128,128)
    
    # export onnx model
    torch.onnx.export(net_G, 
                      input, 
                      './teeth_enhance.onnx', 
                      export_params=True, 
                      do_constant_folding=True, 
                      opset_version=12, 
                      input_names=['lq_img'], 
                      output_names=['hq_img'],
                      dynamic_axes={'input': {0: 'batch'}}
                      )
    
    print("ONNX model export sucessfully!")
    
    

if __name__ == '__main__':
    main()