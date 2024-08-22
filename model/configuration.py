import torch
import time, os 
import datetime
from os.path import join
from copy import deepcopy
from data.vqgan_dataset import Datagan_vqgan
from model import VQAutoEncoder, VQGANDiscriminator
from runer.loss import *
from utils.loss_util import *
from tensorboardX import SummaryWriter

class Model_config():
    def __init__(self, opt, mode):
        self.opt = opt
        self.mode = mode
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def build_net(self):
        net_g_opt = self.opt['network_g']
        net_d_opt = self.opt['network_d']
        # set model
        if self.mode == 'vqgan':
            self.net_G = VQAutoEncoder(img_size=net_g_opt['img_size'],
                                       nf=net_g_opt['nf'],
                                       ch_mult=net_g_opt['ch_mult'],
                                       codebook_size=net_g_opt['codebook_size']).to(self.device)
            
            self.net_D = VQGANDiscriminator(nc=net_d_opt['nc'],
                                            ndf=net_d_opt['ndf']).to(self.device)
            
    def build_data(self):
        # set data
        if self.mode == 'vqgan':
            self.train_dataset = Datagan_vqgan(self.opt)
            self.train_data_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.opt['dataset'].get('batch_size'),
                shuffle=True,
                drop_last=False,
                num_workers=self.opt['dataset'].get('num_workers'),
                pin_memory=False
            )
            
    def build_optimizers(self):
        self.optimizers = []
        train_opt = self.opt['train']
        # optimizer g
        optim_params_g = []
        for k, v in self.net_G.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                print(f'Params {k} will not be optimized.')
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_D.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)
        
    def build_loss(self):
        train_opt = self.opt['train']
        # define losses
        if train_opt.get('pixel_loss'):
            self.cri_pix = L1Loss(train_opt['pixel_loss'].get('loss_weight')).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_loss'):
            self.cri_perceptual = LPIPSLoss(train_opt['perceptual_loss'].get('loss_weight')).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_loss'):
            self.cri_gan = GANLoss(gan_type=train_opt['gan_loss'].get('gan_type'),
                                   loss_weight=train_opt['gan_loss'].get('loss_weight')).to(self.device)

        if train_opt.get('codebook_loss'):
            self.l_weight_codebook = train_opt['codebook_loss'].get('loss_weight', 1.0)
        else:
            self.l_weight_codebook = 1.0
        
        self.vqgan_quantizer = self.opt['network_g']['quantizer']

        self.net_g_start_iter = train_opt.get('net_g_start_iter', 0)
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_start_iter = train_opt.get('net_d_start_iter', 0)
        self.disc_weight = train_opt.get('disc_weight', 0.8)
        
    
    def set_data(self, data):
        self.input = data['in'].to(self.device)
        
    
    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supperted yet.')
        return optimizer
    
    def calculate_adaptive_weight(self, recon_loss, g_loss, last_layer, disc_weight_max):
        recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
        return d_weight
    
    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight
    
    def create_exppath(self):
        """ init experiment path to save log, ckpt ect.
        """
        project_name = self.opt['project_name']
        
        cur_path = os.getcwd()
        cur_date = datetime.date.today()
        cur_time = time.strftime("%H:%M:%S")
        
        exp_path = join(cur_path, 'exp')
        os.makedirs(exp_path, exist_ok=True)
        
        self.project_path = join(exp_path, str(cur_date) + '_' + str(cur_time) + '_' + project_name)
        os.makedirs(self.project_path, exist_ok=True)
        
        self.log_path = join(self.project_path, 'log')
        self.vis_path = join(self.project_path, 'vis')
        self.model_path = join(self.project_path, 'ckpt')
        
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.vis_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        
        
    def resume_state(self):
        """ reload pretrained model.
        """
        resume_path = self.opt['path']
        
        if resume_path.get('net_g'):
            self.load_network(self.net_G, resume_path.get('net_g'))
            self.load_network(self.net_D, resume_path.get('net_d'))
            
            self.net_G.train()
            self.net_D.train()
            
            self.project_path = resume_path.get('net_g').split('ckpt')[0]
            self.log_path = join(self.project_path, 'log')
            self.vis_path = join(self.project_path, 'vis')
            self.model_path = join(self.project_path, 'ckpt')
        else:
            self.create_exppath()
        
        # log
        self.writer = SummaryWriter(self.log_path)
            
    def load_network(self, net, load_path, strict=True, param_key='params'):
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