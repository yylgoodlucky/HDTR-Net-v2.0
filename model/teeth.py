import torch
import torch.nn as nn
from numpy import random

from utils.model_util import *
from model import VQAutoEncoder, ResBlock

class StyleGAN2GeneratorClean(nn.Module):
    """Clean version of StyleGAN2 Generator.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        narrow (float): Narrow ratio for channels. Default: 1.0.
    """

    def __init__(self, out_size, num_style_feat=256, num_mlp=8, channel_multiplier=2, narrow=1):
        super(StyleGAN2GeneratorClean, self).__init__()
        # Style MLP layers
        self.num_style_feat = num_style_feat
        style_mlp_layers = [NormStyleCode()]
        for i in range(num_mlp):
            style_mlp_layers.extend(
                [nn.Linear(num_style_feat, num_style_feat, bias=True),
                 nn.LeakyReLU(negative_slope=0.2, inplace=True)])
        self.style_mlp = nn.Sequential(*style_mlp_layers)
        # initialization
        self.style_mlp.apply(custom_init_weights_leaky_relu)

        # channel list
        channels = {
            '4': int(512 * narrow),
            '8': int(512 * narrow),
            '16': int(256 * narrow),
            '32': int(256 * narrow),
            '64': int(64 * channel_multiplier * narrow),
            '128': int(64 * channel_multiplier * narrow),
            '256': int(32 * channel_multiplier * narrow),
            '512': int(32 * channel_multiplier * narrow),
            '1024': int(8 * channel_multiplier * narrow)
        }
        self.channels = channels
        
        self.constant_input = ConstantInput(channels['4'], size=4)
        self.style_conv1 = StyleConv(
            channels['4'],
            channels['4'],
            kernel_size=3,
            num_style_feat=num_style_feat,
            demodulate=True,
            sample_mode=None)
        self.to_rgb1 = ToRGB(channels['4'], num_style_feat, upsample=False)

        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size) * 2
        self.num_latent = self.log_size * 2 - 2

        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channels = channels['4']
        # noise
        for layer_idx in range(self.num_layers):
            resolution = 2**((layer_idx + 5) // 2)
            shape = [1, 1, resolution, resolution]
            self.noises.register_buffer(f'noise{layer_idx}', torch.randn(*shape))
            
        # style convs and to_rgbs
        for i in range(3, self.log_size + 3):
            out_channels = channels[f'{2**i}']
            if i == 3:
                self.style_convs.append(
                    StyleConv(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        num_style_feat=num_style_feat,
                        demodulate=True,
                        sample_mode=None))
                self.to_rgbs.append(ToRGB(out_channels, num_style_feat, upsample=False))
            else:
                self.style_convs.append(
                    StyleConv(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        num_style_feat=num_style_feat,
                        demodulate=True,
                        sample_mode='upsample'))
                self.to_rgbs.append(ToRGB(out_channels, num_style_feat, upsample=True))
            self.style_convs.append(
                StyleConv(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode=None))
            
            in_channels = out_channels
        
        self.final_tconv = StyleConv(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        num_style_feat=num_style_feat,
                        demodulate=True,
                        sample_mode='upsample')

    def make_noise(self):
        """Make noise for noise injection."""
        device = self.constant_input.weight.device
        noises = [torch.randn(1, 1, 4, 4, device=device)]

        for i in range(3, self.log_size + 3):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=device))

        return noises

    def get_latent(self, x):
        return self.style_mlp(x)

    def mean_latent(self, num_latent):
        latent_in = torch.randn(num_latent, self.num_style_feat, device=self.constant_input.weight.device)
        latent = self.style_mlp(latent_in).mean(0, keepdim=True)
        return latent

    def forward(self,
                styles,
                input_is_latent=False,
                noise=None,
                randomize_noise=True,
                truncation=1,
                truncation_latent=None,
                inject_index=None,
                return_latents=False):
        """Forward function for StyleGAN2GeneratorClean.

        Args:
            styles (list[Tensor]): Sample codes of styles.
            input_is_latent (bool): Whether input is latent style. Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
            truncation (float): The truncation ratio. Default: 1.
            truncation_latent (Tensor | None): The truncation latent tensor. Default: None.
            inject_index (int | None): The injection index for mixing noise. Default: None.
            return_latents (bool): Whether to return style latents. Default: False.
        """
        # style codes -> latents with Style MLP layer
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]
        # style truncation
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_truncation
        # get style latents with injection
        if len(styles) == 1:
            inject_index = self.num_latent

            if styles[0].ndim < 3:
                # repeat latent code for all the layers
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:  # used for encoder with different latent code for each layer
                latent = styles[0]
        elif len(styles) == 2:  # mixing noises
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)

        # main generation
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2],
                                                        noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)  # feature back to the rgb space
            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None

class StyleGAN2GeneratorCSFT(StyleGAN2GeneratorClean):
    """StyleGAN2 Generator with SFT modulation (Spatial Feature Transform).

    It is the clean version without custom compiled CUDA extensions used in StyleGAN2.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        narrow (float): The narrow ratio for channels. Default: 1.
        sft_half (bool): Whether to apply SFT on half of the input channels. Default: False.
    """

    def __init__(self, out_size, num_style_feat=256, num_mlp=8, channel_multiplier=2, narrow=1, sft_half=False):
        super(StyleGAN2GeneratorCSFT, self).__init__(
            out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            narrow=narrow)
        self.sft_half = sft_half

    def forward(self,
                styles,
                conditions,
                input_is_latent=False,
                noise=None,
                randomize_noise=True,
                truncation=1,
                truncation_latent=None,
                inject_index=None,
                return_latents=False):
        """Forward function for StyleGAN2GeneratorCSFT.

        Args:
            styles (list[Tensor]): Sample codes of styles.
            conditions (list[Tensor]): SFT conditions to generators.
            input_is_latent (bool): Whether input is latent style. Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
            truncation (float): The truncation ratio. Default: 1.
            truncation_latent (Tensor | None): The truncation latent tensor. Default: None.
            inject_index (int | None): The injection index for mixing noise. Default: None.
            return_latents (bool): Whether to return style latents. Default: False.
        """
        # style codes -> latents with Style MLP layer
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]
        # style truncation
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_truncation
        # get style latents with injection
        if len(styles) == 1:
            inject_index = self.num_latent

            if styles[0].ndim < 3:
                # repeat latent code for all the layers
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:  # used for encoder with different latent code for each layer
                latent = styles[0]
        elif len(styles) == 2:  # mixing noises
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)

        # main generation
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2],
                                                        noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)

            # the conditions may have fewer levels
            if i <= len(conditions):
                # SFT part to combine the conditions
                if self.sft_half:  # only apply SFT to half of the channels
                    out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
                    print(i, out_sft.shape, conditions[i].shape)

                    out_sft = out_sft * conditions[i - 1] + conditions[i]
                    out = torch.cat([out_same, out_sft], dim=1)
                else:  # apply SFT to all the channels
                    out = out * conditions[i - 1] + conditions[i]

            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)  # feature back to the rgb space
            i += 2

        image = skip
        if return_latents:
            return image, latent
        else:
            return image, None

class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt_q, tgt_k,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        # self attention
        tgt = tgt_q
        tgt_q = self.norm1(tgt_q)
        tgt_k = self.norm1(tgt_k)
        q = self.with_pos_embed(tgt_q, query_pos)
        k = self.with_pos_embed(tgt_k, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt_q, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt

class Teeth_Enhance(VQAutoEncoder):
    def __init__(self, opt):
        self.opt = opt

        self.in_channels = opt['in_channels']
        self.nf = opt['nf']
        self.n_blocks = opt['nblocks']
        self.embed_dim = opt['embed_dim']
        self.ch_mult = opt['ch_mult']
        self.resolution = opt['resolution']
        self.attn_resolutions = opt['attn_resolutions']
        self.codebook_size = opt['codebook_size']
        super(Teeth_Enhance, self).__init__(self.resolution, self.nf, self.ch_mult, 'nearest', self.n_blocks, self.attn_resolutions, self.codebook_size)
        
        # cross attention for better performance Configuration Hyperparameters
        self.latent_size = opt['latent_size']
        self.dim_emb = opt['dim_emb']
        self.dim_mlp = self.dim_emb*2
        n_head = opt['n_head']
        self.n_layers = opt['n_layers']
        self.input_is_latent = opt['input_is_latent']
        
        # SFT for condition fusion Configuration Hyperparameters
        out_size = opt['out_size']
        num_style_feat = opt['num_style_feat']
        num_mlp = opt['num_mlp']
        channel_multiplier = opt['channel_multiplier']
        narrow = opt['narrow']
        sft_half = opt['sft_half']
        
        channels = {'1': 256, '2': 128, '4': 128, '8': 64, '16': 32, '32': 16, '64': 3}
        
        # after second residual block for > 16, before attn layer for ==16
        connect_list=['16', '32', '64', '128', '256', '512']
        self.connect_list = connect_list
        self.fuse_encoder_block = {'512':2, '256':5, '128':8, '64':11, '32':14, '16':18}
        
        self.position_emb = nn.Parameter(torch.zeros(self.latent_size, self.dim_emb))
        self.feat_emb = nn.Linear(256, self.dim_emb)
        
        # transformer
        self.ft_layers = nn.Sequential(*[TransformerSALayer(embed_dim=self.dim_emb, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0) for _ in range(self.n_layers)])
        
        channels = {'1': 256, '2': 128, '4': 128, '8': 64, '16': 64, '32': 32, '64': 32}
        # upsample
        in_channels = channels['1']
        self.conv_body_up = nn.ModuleList()
        for i in range(0, len(self.connect_list)):
            out_channels = channels[f'{2**i}']
            if i == 0:
                out_channels = in_channels
                self.conv_body_up.append(ResBlock(in_channels, out_channels))
            else:
                self.conv_body_up.append(ResBlock(in_channels, out_channels, mode='up'))
                in_channels = out_channels
        
        # for SFT modulations (scale and shift)
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(0, len(self.connect_list)):
            out_channels = channels[f'{2**i}']
            if sft_half:
                sft_out_channels = out_channels
            else:
                sft_out_channels = out_channels * 2
                
            self.condition_scale.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1)))
            self.condition_shift.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1)))
            
            
        # the decoder: stylegan2 generator with SFT modulations
        self.stylegan_decoder = StyleGAN2GeneratorCSFT(
            out_size=out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            narrow=narrow,
            sft_half=sft_half)
        
    def forward(self, x, return_latents=False, randomize_noise=True):
        b = x.shape[0]
        #################### Encoder #####################
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):   # shape: [b,256,4,4] len_block: 25
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()

        #################### Transformer #####################
        lq_feat = x
        pos_emb = self.position_emb.unsqueeze(1).repeat(1,x.shape[0],1)
        feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2,0,1))
        query_emb = feat_emb
        
        # codebook query
        output, _, quant_stats = self.quantize(x)
        min_encoding_indices = quant_stats['min_encoding_indices']
        codebook = min_encoding_indices.view(b, -1)
        key_emb = self.quantize.get_codebook_feat(codebook, shape=[b,4,4,self.dim_emb])
        key_emb = key_emb.view(-1, b, self.dim_emb)
        
        # Transformer encoder
        for layer in self.ft_layers:
            style_code = layer(query_emb, key_emb, query_pos=pos_emb)   # 16,b,256
        
        #################### Decode #####################
        conditions = []
        feat = lq_feat
        for i in range(len(out_list)):
            # add unet skip
            feat = feat + enc_feat_dict[str(feat.shape[-1])]
            # ResUpLayer
            feat = self.conv_body_up[i](feat)
            # generate scale and shift for SFT layers
            scale = self.condition_scale[i](feat)
            conditions.append(scale.clone())
            shift = self.condition_shift[i](feat)
            conditions.append(shift.clone())
        

        #################### Stylegan #####################
        style_code = style_code.permute(1, 0, 2)
        image, _ = self.stylegan_decoder([style_code],
                                         conditions,
                                         return_latents=return_latents,
                                         input_is_latent=self.input_is_latent,
                                         randomize_noise=randomize_noise)
        return image    


if __name__=="__main__":
    input = torch.randn(1, 3, 128 ,128)
    net = Teeth_Enhance()
    output = net(input)
    print(output.shape)
