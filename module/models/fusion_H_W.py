import torch
import torch.nn.functional as F
import torch.nn as nn
from operator import itemgetter
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states

EPSILON = 1e-10
class AdaptiveFusion(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.weight1 = nn.Parameter(torch.tensor(1.0))
        self.weight2 = nn.Parameter(torch.tensor(1.0))
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )


        self.spatial_att = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):

        if x1.size(1) != x2.size(1):
            min_channels = min(x1.size(1), x2.size(1))
            x1 = x1[:, :min_channels]
            x2 = x2[:, :min_channels]


        combined = x1 + x2


        channel_att = self.channel_att(combined)


        spatial_avg = torch.mean(combined, dim=1, keepdim=True)
        spatial_max, _ = torch.max(combined, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([spatial_avg, spatial_max], dim=1))


        attention = channel_att * spatial_att


        w1 = torch.sigmoid(self.weight1)
        w2 = torch.sigmoid(self.weight2)


        fused = w1 * x1 * attention + w2 * x2 * attention

        return fused


def optimized_fusion(tensor1, tensor2):
    orig_shape1 = tensor1.shape
    orig_shape2 = tensor2.shape
    has_time_dim = False

    if tensor1.dim() == 6:
        B, C, D, H, W, T = tensor1.shape
        tensor1 = tensor1.permute(0, 5, 1, 2, 3, 4).contiguous().view(B * T, C, D, H, W)
        has_time_dim = True
        time_dim = T

    if tensor2.dim() == 6:
        B, C, D, H, W, T = tensor2.shape
        tensor2 = tensor2.permute(0, 5, 1, 2, 3, 4).contiguous().view(B * T, C, D, H, W)
        has_time_dim = True
        time_dim = T

    if tensor1.size(0) != tensor2.size(0):
        min_batch = min(tensor1.size(0), tensor2.size(0))
        tensor1 = tensor1[:min_batch]
        tensor2 = tensor2[:min_batch]

    device = tensor1.device

    in_channels = min(tensor1.size(1), tensor2.size(1))
    fusion_module = AdaptiveFusion(in_channels).to(device)

    fused = fusion_module(tensor1, tensor2)

    return fused

def attention_fusion_channel(tensor1, tensor2, p_type):
    f_channel = channel_fusion(tensor1, tensor2, p_type)
    tensor_f = f_channel[0]
    tensor_f1 = f_channel[1]
    tensor_f2 = f_channel[2]
    return tensor_f, tensor_f1, tensor_f2


def channel_fusion(tensor1, tensor2, p_type):
    shape = tensor1.size()
    global_p1 = channel_attention(tensor1, p_type)
    global_p2 = channel_attention(tensor2, p_type)
    global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
    global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)
    global_p_w1 = global_p_w1.repeat(1, 1, *shape[2:])
    global_p_w2 = global_p_w2.repeat(1, 1, *shape[2:])
    tensor_f1 = 0.8 * global_p_w1 * tensor1
    tensor_f2 = 0.8 * global_p_w2 * tensor2
    tensor_f = tensor_f1 + tensor_f2
    return tensor_f, tensor_f1, tensor_f2


def channel_attention(tensor, pooling_type='avg'):
    shape = tensor.size()
    pooling_function = F.avg_pool3d

    if pooling_type == 'attention_avg':
        pooling_function = F.avg_pool3d
    elif pooling_type == 'attention_max':
        pooling_function = F.max_pool3d
    elif pooling_type == 'attention_nuclear':
        return nuclear_pooling(tensor)

    global_p = pooling_function(tensor, kernel_size=shape[2:])
    return global_p


def nuclear_pooling(tensor, kernel_size=None):
    shape = tensor.size()
    vectors = torch.zeros(1, shape[1], 1, 1, 1, device=tensor.device)

    for i in range(shape[1]):
        slice_3d = tensor[0, i, ...]
        flattened = slice_3d.flatten()
        u, s, v = torch.linalg.svd(flattened, full_matrices=False)
        s_sum = torch.sum(s)
        vectors[0, i, 0, 0, 0] = s_sum

    return vectors

class AxialAttentionTransformer(nn.Module):
    def __init__(self, dim, depth, heads=8, dim_heads=None, dim_index=1, reversible=True, axial_pos_emb_shape=None):
        super().__init__()
        permutations = calculate_permutations(3, dim_index)

        get_ff = lambda: nn.Sequential(
            ChanLayerNorm(dim),
            nn.Conv2d(dim, dim * 4, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim * 4, dim, 3, padding=1)
        )

        self.pos_emb = AxialPositionalEmbedding(dim, axial_pos_emb_shape, dim_index) if exists(
            axial_pos_emb_shape) else nn.Identity()

        blocks = []
        for _ in range(depth):
            attn_functions = nn.ModuleList(
                [PermuteToFrom(permutation, PreNorm(dim, SelfAttention(dim, heads, dim_heads))) for permutation in
                 permutations])
            conv_functions = nn.ModuleList([get_ff(), get_ff()])

            for attn_fn in attn_functions:
                blocks.append((attn_fn, conv_functions[0]))
                blocks.append((attn_fn, conv_functions[1]))

        execute_type = ReversibleSequence if reversible else Sequential
        self.layers = execute_type(blocks)

    def forward(self, x):
        x = self.pos_emb(x)
        return self.layers(x)


class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)


class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1, y2 = None, None

        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)

        return torch.cat([y1, y2], dim=1)

    def backward_pass(self, y, dy, f_args={}, g_args={}):
        y1, y2 = torch.chunk(y, 2, dim=1)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=1)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=1)
            dx = torch.cat([dx1, dx2], dim=1)

        return x, dx


class IrreversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, x, f_args, g_args):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1 = x1 + self.f(x2, **f_args)
        y2 = x2 + self.g(y1, **g_args)
        return torch.cat([y1, y2], dim=1)


class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        kwargs = ctx.kwargs
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None


class ReversibleSequence(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for (f, g) in blocks])

    def forward(self, x, arg_route=(True, True), **kwargs):
        f_args, g_args = map(lambda route: kwargs if route else {}, arg_route)
        block_kwargs = {'f_args': f_args, 'g_args': g_args}
        x = torch.cat((x, x), dim=1)
        x = _ReversibleFunction.apply(x, self.blocks, block_kwargs)
        return torch.stack(x.chunk(2, dim=1)).mean(dim=0)


def exists(val):
    return val is not None


def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))


def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)

def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)

    return permutations


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x)
            x = x + g(x)
        return x


class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape

        axial = axial.reshape(-1, t, d)

        axial = self.fn(axial, **kwargs)


        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape, emb_dim_index=1):
        super().__init__()
        self.shape = shape
        self.emb_dim_index = emb_dim_index
        self.parameters = nn.ParameterList()
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]

        for i, axial_dim in enumerate(shape):
            param_shape = [1] * total_dimensions
            param_shape[emb_dim_index] = dim
            param_shape[ax_dim_indexes[i]] = axial_dim
            param = nn.Parameter(torch.randn(*param_shape))
            self.parameters.append(param)

    def forward(self, x):
        for param in self.parameters:
            if param.dim() != x.dim():
                raise ValueError(f"Position encoding dim {param.dim()} does not match input dim {x.dim()}")
            x = x + param
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads=None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out