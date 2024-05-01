import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from .vision_base import VisionBase


class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs
    
class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(
                                dim, heads=heads, dropout_rate=attn_dropout_rate
                            ),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
        self.net = IntermediateSequential(*layers)

    def forward(self, x):
        return self.net(x)

class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=5000):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return x + position_embeddings

class SegmentationTransformer(VisionBase):
    def __init__(self, params, device):
        super(SegmentationTransformer, self).__init__(params, device)

        self.input_size = params['input_size']
        self.output_size = params['output_size']
        self.embedding_dim = params['embedding_dim']
        self.num_heads = params['num_heads']
        self.patch_dim = params['patch_dim']
        self.num_channels = params['num_channels']
        self.dropout_rate = params['dropout_rate']
        self.attn_dropout_rate = params['attn_dropout_rate']
        self.conv_patch_representation = params['conv_patch_representation']
        self.positional_encoding_type = params['positional_encoding_type']
        self.num_layers = params['num_layers']
        self.hidden_dim = params['hidden_dim']
        self.criterion = params['criterion']
        self.aux_layers = params.get('aux_layers', None)
        self.out = None

        assert self.embedding_dim % self.num_heads == 0
        assert self.input_size[0] % self.patch_dim == 0
        assert self.input_size[1] % self.patch_dim == 0

        self.num_patches = int((self.input_size[0] // self.patch_dim) * (self.input_size[1] // self.patch_dim))
        self.seq_length = self.num_patches
        self.flatten_dim = self.patch_dim * self.patch_dim * self.num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if self.positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif self.positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            self.embedding_dim,
            self.num_layers,
            self.num_heads,
            self.hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(self.embedding_dim)

        if self.conv_patch_representation:
            self.conv_x = nn.Conv2d(
                self.num_channels,
                self.embedding_dim,
                kernel_size=(self.patch_dim, self.patch_dim),
                stride=(self.patch_dim, self.patch_dim),
                padding=self._get_padding(
                    'VALID', (self.patch_dim, self.patch_dim),
                ),
            )
        else:
            self.conv_x = None
        
        self.to(device = device)

    def _init_decode(self):
        raise NotImplementedError("Should be implemented in child class!!")

    def encode(self, x):
        n, c, h, w = x.shape
        if self.conv_patch_representation:
            # combine embedding w/ conv patch distribution
            x = self.conv_x(x)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.embedding_dim)
        else:
            x = (
                x.unfold(2, self.patch_dim, self.patch_dim)
                .unfold(3, self.patch_dim, self.patch_dim)
                .contiguous()
            )
            x = x.view(n, c, -1, self.patch_dim ** 2)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
            x = self.linear_encoding(x)

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)

        return x, intmd_x

    def decode(self, x):
        raise NotImplementedError("Should be implemented in child class!!")
    
    def get_feature_map(self, x, auxillary_output_layers):
        encoder_output, intmd_encoder_outputs = self.encode(x['image'])
        auxillary_output_layers = self.aux_layers
        feature_map = self.decode(
            encoder_output, intmd_encoder_outputs, auxillary_output_layers
        )
        return feature_map, intmd_encoder_outputs
        
    
    def forward(self, x, auxillary_output_layers=None, return_loss=False):
        x['image'] = T.Resize(self.input_size, interpolation=T.InterpolationMode.NEAREST)(x['image'])
        feature_map, intmd_encoder_outputs = self.get_feature_map(x, auxillary_output_layers)
        result = self.out(feature_map)
        result = T.Resize(self.output_size, interpolation=T.InterpolationMode.NEAREST)(result)

        if auxillary_output_layers is not None:
            auxillary_outputs = {}
            for i in auxillary_output_layers:
                val = str(2 * i - 1)
                _key = 'Z' + str(i)
                auxillary_outputs[_key] = intmd_encoder_outputs[val]
        
        if return_loss:
            gt = x['gt']
            loss = self.criterion(result.sigmoid(), gt)
            return result, loss
        else:
            x['pred'] = result.sigmoid()
            return x

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.input_size[0] / self.patch_dim),
            int(self.input_size[1] / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class SETR_Naive(SegmentationTransformer):
    def __init__(self, params, device):

        SegmentationTransformer.__init__(self, params, device)
        

        self.num_classes = params['num_classes']
        self._init_decode()

        self.to(device = device)

    def _init_decode(self):
        self.conv1 = nn.Conv2d(
            in_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1,
            padding=self._get_padding('VALID', (1, 1),),
        )
        self.bn1 = nn.BatchNorm2d(self.embedding_dim)
        self.act1 = nn.ReLU()
        
        self.out = IntermediateSequential(return_intermediate=False)
        self.out.add_module(
            "Conv2d",
            nn.Conv2d(
                in_channels=self.embedding_dim,
                out_channels=self.num_classes,
                kernel_size=1,
                stride=1,
                padding=self._get_padding('VALID', (1, 1),),
            )
        ) 
        self.out.add_module(
            "Upsample",
            nn.Upsample(
                scale_factor=self.patch_dim, mode='bilinear'
            )
        )

    def decode(self, x, intmd_x, intmd_layers=None):
        x = self._reshape_output(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

class SETR_MLA(SegmentationTransformer,):
    def __init__(self, params, device):
        SegmentationTransformer.__init__(self, params, device)

        self.num_classes = params['num_classes']
        self._init_decode()

        self.to(device = device)

    def _init_decode(self):
        self.net1_in, self.net1_intmd, self.net1_out = self._define_agg_net()
        self.net2_in, self.net2_intmd, self.net2_out = self._define_agg_net()
        self.net3_in, self.net3_intmd, self.net3_out = self._define_agg_net()
        self.net4_in, self.net4_intmd, self.net4_out = self._define_agg_net()

        # fmt: off
        self.out = IntermediateSequential(return_intermediate=False)
        self.out.add_module(
            "conv_1",
            nn.Conv2d(
                in_channels=self.embedding_dim, out_channels=self.num_classes,
                kernel_size=1, stride=1,
                padding=self._get_padding('VALID', (1, 1),),
            )
        )
        self.out.add_module(
            "upsample_1",
            nn.Upsample(scale_factor=4, mode='bilinear')
        )
        # fmt: on

    def decode(self, x, intmd_x, intmd_layers=None):
        assert intmd_layers is not None, "pass the intermediate layers for MLA"

        encoder_outputs = {}
        all_keys = []
        for i in intmd_layers:
            val = str(2 * i - 1)
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = intmd_x[val]
        all_keys.reverse()

        temp_x = encoder_outputs[all_keys[0]]
        temp_x = self._reshape_output(temp_x)
        key0_intmd_in = self.net1_in(temp_x)
        key0_out = self.net1_out(key0_intmd_in)

        temp_x = encoder_outputs[all_keys[1]]
        temp_x = self._reshape_output(temp_x)
        key1_in = self.net2_in(temp_x)
        key1_intmd_in = key1_in + key0_intmd_in
        key1_intmd_out = self.net2_intmd(key1_intmd_in)
        key1_out = self.net2_out(key1_intmd_out)

        temp_x = encoder_outputs[all_keys[2]]
        temp_x = self._reshape_output(temp_x)
        key2_in = self.net3_in(temp_x)
        key2_intmd_in = key2_in + key1_intmd_in
        key2_intmd_out = self.net3_intmd(key2_intmd_in)
        key2_out = self.net3_out(key2_intmd_out)

        temp_x = encoder_outputs[all_keys[3]]
        temp_x = self._reshape_output(temp_x)
        key3_in = self.net4_in(temp_x)
        key3_intmd_in = key3_in + key2_intmd_in
        key3_intmd_out = self.net4_intmd(key3_intmd_in)
        key3_out = self.net4_out(key3_intmd_out)

        out = torch.cat((key0_out, key1_out, key2_out, key3_out), dim=1)
        return out

    # fmt: off
    def _define_agg_net(self):
        model_in = IntermediateSequential(return_intermediate=False)
        model_in.add_module(
            "layer_1",
            nn.Conv2d(
                self.embedding_dim, int(self.embedding_dim / 2), 1, 1,
                padding=self._get_padding('VALID', (1, 1),),
            ),
        )

        model_intmd = IntermediateSequential(return_intermediate=False)
        model_intmd.add_module(
            "layer_intmd",
            nn.Conv2d(
                int(self.embedding_dim / 2), int(self.embedding_dim / 2), 3, 1,
                padding=self._get_padding('SAME', (3, 3),),
            ),
        )

        model_out = IntermediateSequential(return_intermediate=False)
        model_out.add_module(
            "layer_2",
            nn.Conv2d(
                int(self.embedding_dim / 2), int(self.embedding_dim / 2), 3, 1,
                padding=self._get_padding('SAME', (3, 3),),
            ),
        )
        model_out.add_module(
            "layer_3",
            nn.Conv2d(
                int(self.embedding_dim / 2), int(self.embedding_dim / 4), 3, 1,
                padding=self._get_padding('SAME', (3, 3),),
            ),
        )
        model_out.add_module(
            "upsample", nn.Upsample(scale_factor=4, mode='bilinear')
        )
        return model_in, model_intmd, model_out
    # fmt: on

class SETR_PUP(SegmentationTransformer):
    def __init__(self, params, device):
        SegmentationTransformer.__init__(self, params, device)

        self.num_classes = params['num_classes']
        self._init_decode()

        self.to(device = device)

    def _init_decode(self):
        extra_in_channels = int(self.embedding_dim / 4)
        in_channels = [
            self.embedding_dim,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
        ]
        out_channels = [
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            self.num_classes,
        ]

        modules = []
        for i, (in_channel, out_channel) in enumerate(
            zip(in_channels, out_channels)
        ):
            modules.append(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=self._get_padding('VALID', (1, 1),),
                )
            )
            if i != 4:
                modules.append(nn.Upsample(scale_factor=2, mode='bilinear'))

        decode_modules = modules[:-1]
        self.decode_net = IntermediateSequential(
            *decode_modules, return_intermediate=False
        )

        out = modules[-1:]
        self.out = IntermediateSequential(
            *out, return_intermediate=False
        )

    def decode(self, x, intmd_x, intmd_layers=None):
        x = self._reshape_output(x)
        x = self.decode_net(x)
        return x