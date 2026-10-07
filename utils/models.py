import torch
from torch.nn import (
    ReLU, Linear, Sequential, Module, ModuleList, Flatten,
    Conv2d, BatchNorm2d, GroupNorm, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d,
)
import math


####################################################################################################
################################################## Quadratic CL model
####################################################################################################

class LinearRegressorCL(Module):
    """Single linear layer for the convex multi-task quadratic CL benchmark.

    The model computes f(x) = w^T x (scalar regression).  The "features"
    sub-module is an empty Sequential (no shared parameters) and "heads" is a
    one-element ModuleList containing the single Linear(d, 1, bias=False) layer.
    This layout mirrors MultiHeadMLP so that code using get_gradient / update
    works unchanged: both iterate over features.parameters() (empty) and then
    heads[task_id].parameters() (w).

    In the quadratic setting all tasks share the same parameter vector w —
    task-specific structure lives entirely in the data distribution
    (QuadraticDataset).  task_id is accepted for interface compatibility but
    always ignored.

    Parameters
    ----------
    input_size : int
        Dimension d of the parameter / feature space.
    init_std : float
        Standard deviation for the Gaussian initialisation of w (default 0.01).
    """

    def __init__(self, input_size: int, init_std: float = 0.01):
        super().__init__()
        self.input_size = input_size
        self.features   = Sequential()                         # empty — no shared params
        self.heads      = ModuleList([Linear(input_size, 1, bias=False)])

        torch.nn.init.normal_(self.heads[0].weight, mean=0.0, std=init_std)

    def forward(self, x, task_id: int = 0, out: str = 'logits'):
        # out argument accepted for interface compatibility; only 'logits' path used
        return self.heads[0](x).squeeze(-1)   # (B,)

    def get_gradient(self, task_id: int = 0) -> torch.Tensor:
        """Return the flattened gradient of the loss w.r.t. w."""
        g = []
        for p in self.features.parameters():          # empty — contributes nothing
            g.append(p.grad.reshape(-1))
        for p in self.heads[0].parameters():          # the weight tensor w
            g.append(p.grad.reshape(-1))
        return torch.cat(g) if g else torch.zeros(self.input_size)

    @torch.no_grad()
    def update(self, update: torch.Tensor, task_id: int = 0,
               lr: float = 1e-3, wd: float = 0.0):
        """Apply a (possibly preconditioned) update: w ← w - lr * update."""
        p = self.heads[0].weight
        if wd > 0:
            p.mul_(1.0 - lr * wd)
        p.add_(update.reshape(p.shape), alpha=-lr)

    def get_w(self) -> torch.Tensor:
        """Return the current weight vector as a 1-D CPU tensor of shape (d,)."""
        return self.heads[0].weight.detach().squeeze(0).cpu()

####################################################################################################
################################################## MLP
####################################################################################################

class MultiHeadMLP(Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads, bias=False):
        super(MultiHeadMLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_heads = num_heads

        self.features = Sequential(*[
            Linear(input_size, hidden_size, bias=bias),
            ReLU(inplace=False),
            Linear(hidden_size, hidden_size, bias=bias),
            ReLU(inplace=False)
        ])
        self.heads = ModuleList([
            Linear(hidden_size, output_size, bias=bias)
            for _ in range(self.num_heads)
        ])

    def forward(self, x, task_id, out='logits'):
        assert out in ['logits', 'features', 'both']
        x = x.view(x.shape[0], -1)

        features = self.features(x)
        if out == 'features':
            return features

        logits = self.heads[task_id](features)
        if out == 'logits':
            return logits

        return logits, features

    def get_gradient(self, task_id):
        g = []
        for p in self.features.parameters():
            g.append(p.grad.reshape(-1))

        for p in self.heads[task_id].parameters():
            g.append(p.grad.reshape(-1))
        return torch.cat(g)

    @torch.no_grad()
    def update(self, update, task_id, lr, wd):
        count = 0
        for p in self.features.parameters():
            if wd > 0 and p.ndim > 1:
                p.mul_(1 - lr * wd)
            u = update[count:count+p.numel()].reshape(p.shape)
            p.add_(u, alpha=-lr)
            count += p.numel()

        for p in self.heads[task_id].parameters():
            if wd > 0 and p.ndim > 1:
                p.mul_(1 - lr * wd)
            u = update[count:count+p.numel()].reshape(p.shape)
            p.add_(u, alpha=-lr)
            count += p.numel()



####################################################################################################
################################################## ResNet for CIFAR-10
####################################################################################################

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(Module):
    expansion = 1

    def __init__(self, block_gates, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.block_gates = block_gates
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu1 = ReLU(inplace=False)  # To enable layer removal inplace must be False
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.relu2 = ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = out = x

        if self.block_gates[0]:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)

        if self.block_gates[1]:
            out = self.conv2(out)
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNetCifar(Module):
    def __init__(self, block, layers, output_size, num_heads):
        self.output_size = output_size
        self.num_heads = num_heads

        self.nlayers = 0
        # Each layer manages its own gates
        self.layer_gates = []
        for layer in range(3):
            # For each of the 3 layers, create block gates: each block has two layers
            self.layer_gates.append([])  # [True, True] * layers[layer])
            for blk in range(layers[layer]):
                self.layer_gates[layer].append([True, True])

        self.inplanes = 16  # 64
        super(ResNetCifar, self).__init__()

        self.features = Sequential(*[
            Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(self.inplanes),
            ReLU(inplace=False),
            self._make_layer(self.layer_gates[0], block, 16, layers[0]),
            self._make_layer(self.layer_gates[1], block, 32, layers[1], stride=2),
            self._make_layer(self.layer_gates[2], block, 64, layers[2], stride=2),
            AvgPool2d(8, stride=1),
            Flatten()
        ])

        self.heads = ModuleList([
            Linear(64 * block.expansion, output_size)
            for _ in range(self.num_heads)
        ])

        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, task_id, out='logits'):
        assert out in ['logits', 'features', 'both']
        features = self.features(x)
        if out == 'features':
            return features

        logits = self.heads[task_id](features)
        if out == 'logits':
            return logits

        return logits, features

    def _make_layer(self, layer_gates, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(layer_gates[0], self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(layer_gates[i], self.inplanes, planes))

        return Sequential(*layers)


def resnet20(**kwargs):
    model = ResNetCifar(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet32(**kwargs):
    model = ResNetCifar(BasicBlock, [5, 5, 5], **kwargs)
    return model

def resnet44(**kwargs):
    model = ResNetCifar(BasicBlock, [7, 7, 7], **kwargs)
    return model

def resnet56(**kwargs):
    model = ResNetCifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


class _Cifar18Block(Module):
    """Mammoth-style BasicBlock (shortcut 1x1, no layer gates)."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=BatchNorm2d):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.relu = ReLU(inplace=False)
        if stride != 1 or in_planes != planes:
            self.shortcut = Sequential(
                Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes),
            )
        else:
            self.shortcut = Sequential()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet18CifarCL(Module):
    """CIFAR ResNet-18 used by Mammoth on seq-cifar* / seq-tinyimg.

    3x3 stride-1 stem, **no maxpool**, nf=64, four stages [2, 2, 2, 2].
    AdaptiveAvgPool so 32x32 and 64x64 both work. This is *not* torchvision's
    ImageNet ResNet-18 (7x7 stride-2 + maxpool), which silently wrecks 32/64px.
    """

    def __init__(self, output_size, num_heads, nf=64, norm='bn'):
        super().__init__()
        if norm == 'bn':
            norm_layer = BatchNorm2d
        elif norm == 'gn':
            norm_layer = lambda c: GroupNorm(num_groups=min(32, c), num_channels=c)
        else:
            raise ValueError(f'Unknown norm {norm!r}')

        self.num_heads = num_heads
        self.in_planes = nf
        layers = [
            conv3x3(3, nf),
            norm_layer(nf),
            ReLU(inplace=False),
            self._make_layer(nf, 2, 1, norm_layer),
            self._make_layer(nf * 2, 2, 2, norm_layer),
            self._make_layer(nf * 4, 2, 2, norm_layer),
            self._make_layer(nf * 8, 2, 2, norm_layer),
            AdaptiveAvgPool2d(1),
            Flatten(),
        ]
        self.features = Sequential(*layers)
        self.heads = ModuleList([
            Linear(nf * 8, output_size) for _ in range(num_heads)
        ])
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, num_blocks, stride, norm_layer):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(_Cifar18Block(self.in_planes, planes, s, norm_layer))
            self.in_planes = planes
        return Sequential(*blocks)

    def forward(self, x, task_id, out='logits'):
        assert out in ('logits', 'features', 'both')
        features = self.features(x)
        if out == 'features':
            return features
        logits = self.heads[task_id](features)
        if out == 'logits':
            return logits
        return logits, features

    def get_gradient(self, task_id):
        g = [p.grad.reshape(-1) for p in self.features.parameters()]
        g += [p.grad.reshape(-1) for p in self.heads[task_id].parameters()]
        return torch.cat(g)

    @torch.no_grad()
    def update(self, update, task_id, lr, wd):
        count = 0
        for p in self.features.parameters():
            if wd > 0 and p.ndim > 1:
                p.mul_(1 - lr * wd)
            u = update[count:count + p.numel()].reshape(p.shape)
            p.add_(u, alpha=-lr)
            count += p.numel()
        for p in self.heads[task_id].parameters():
            if wd > 0 and p.ndim > 1:
                p.mul_(1 - lr * wd)
            u = update[count:count + p.numel()].reshape(p.shape)
            p.add_(u, alpha=-lr)
            count += p.numel()

####################################################################################################
################################################## LeNet for MNIST and CIFAR-10
####################################################################################################

class LeNet5(Module):
    """
    From https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/
    """
    def __init__(self, input_channels, padding_conv1, output_size, num_heads, hidden_size=None):
        super(LeNet5, self).__init__()
        self.output_size = output_size
        self.num_heads = num_heads

        self.features = Sequential(*[
            Conv2d(input_channels, 6, 5, 1, padding_conv1),
            BatchNorm2d(6),
            ReLU(inplace=False),
            MaxPool2d(2, 2),
            Conv2d(6, 16, 5),
            BatchNorm2d(16),
            ReLU(inplace=False),
            MaxPool2d(2, 2),
            Flatten(),
        ])

        if hidden_size is None:
            self.heads = ModuleList([
                Sequential(*[
                    Linear(16 * 5 * 5, 120),
                    ReLU(inplace=False),
                    Linear(120, 84),
                    ReLU(inplace=False),
                    Linear(84, output_size),
                ])
                for _ in range(self.num_heads)
            ])
        else:
            # custom LeNet with specified hidden_size instead of 120 and 84
            self.heads = ModuleList([
                Sequential(*[
                    Linear(16 * 5 * 5, hidden_size),
                    ReLU(inplace=False),
                    Linear(hidden_size, hidden_size),
                    ReLU(inplace=False),
                    Linear(hidden_size, output_size),
                ])
                for _ in range(self.num_heads)
            ])

    def forward(self, x, task_id, out='logits'):
        assert out in ['logits', 'features', 'both']
        features = self.features(x)
        if out == 'features':
            return features

        logits = self.heads[task_id](features)
        if out == 'logits':
            return logits

        return logits, features

class LeNet5_MNIST(LeNet5):
    def __init__(self, output_size, num_heads, hidden_size=None):
        super(LeNet5_MNIST, self).__init__(input_channels=1, padding_conv1='same', output_size=output_size, num_heads=num_heads, hidden_size=hidden_size)


class LeNet5_CIFAR(LeNet5):
    def __init__(self, output_size, num_heads, hidden_size=None):
        super(LeNet5_CIFAR, self).__init__(input_channels=3, padding_conv1='same', output_size=output_size, num_heads=num_heads, hidden_size=hidden_size)


####################################################################################################
################################################## Conv nets for 224x224 (CLEAR from raw images)
####################################################################################################
#
# Both classes below implement the same CL interface as MultiHeadMLP / ResNetCifar / LeNet5:
#   forward(x, task_id, out='logits' | 'features' | 'both')
#   get_gradient(task_id)  -> flat tensor of all features + heads[task_id] grads
#   update(update, task_id, lr, wd) -> inverse operation
#
# Design notes:
#   * Separate `features` (backbone) and `heads` (ModuleList[Linear]) to match the
#     flatten/unflatten contract expected by DenseCoreMFAC.
#   * CLEAR is domain-incremental, so in practice num_heads=1 (single shared head)
#     regardless of args.use_multihead_model; we still support num_heads>1 for symmetry.
#   * GroupNorm is the default norm layer because BatchNorm running-stats drift
#     across CLEAR buckets confounds any measurement of the optimizer's effect on
#     forgetting (the stats change even under a fixed weight update).


def _make_norm_layer(kind):
    """Factory returning a callable ``num_channels -> norm_module``.

    ``kind='bn'`` -> BatchNorm2d (needed for loading torchvision pretrained weights)
    ``kind='gn'`` -> GroupNorm with 32 groups (capped at num_channels to avoid
                     invalid configs on the stem's 16-channel tensor)
    """
    if kind == 'bn':
        return lambda c: BatchNorm2d(c)
    if kind == 'gn':
        return lambda c: GroupNorm(num_groups=min(32, c), num_channels=c)
    raise ValueError(f'Unknown norm kind: {kind!r}')


class ResNet18CL(Module):
    """torchvision ResNet-18 wrapped to satisfy the CL-model interface.

    Parameters
    ----------
    output_size : int
        Number of logits per head (e.g. 11 for CLEAR10, 101 for CLEAR100).
    num_heads : int
        1 for domain-incremental (CLEAR), N for task-incremental.
    pretrained : bool
        If True, load ImageNet weights (requires norm='bn' because torchvision's
        pretrained weights assume BatchNorm). Only the backbone is pretrained;
        the heads are always freshly initialized.
    norm : {'bn', 'gn'}
        Normalization layer for the backbone. 'gn' is recommended for CL.
    """

    def __init__(self, output_size, num_heads, pretrained=False, norm='gn'):
        super().__init__()
        import torchvision.models as tvm

        if pretrained and norm != 'bn':
            raise ValueError(
                "ResNet-18 pretrained weights are only distributed for BatchNorm; "
                "use norm='bn' together with pretrained=True, or norm='gn' with "
                "pretrained=False."
            )

        norm_layer = _make_norm_layer(norm)
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tvm.resnet18(weights=weights, norm_layer=norm_layer)

        self.features = Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            backbone.avgpool, Flatten(),
        )
        self.feat_dim = backbone.fc.in_features  # 512 for ResNet-18
        self.heads = ModuleList([
            Linear(self.feat_dim, output_size) for _ in range(num_heads)
        ])

    def forward(self, x, task_id, out='logits'):
        assert out in ('logits', 'features', 'both')
        features = self.features(x)
        if out == 'features':
            return features
        logits = self.heads[task_id](features)
        if out == 'logits':
            return logits
        return logits, features

    def get_gradient(self, task_id):
        g = [p.grad.reshape(-1) for p in self.features.parameters()]
        g += [p.grad.reshape(-1) for p in self.heads[task_id].parameters()]
        return torch.cat(g)

    @torch.no_grad()
    def update(self, update, task_id, lr, wd):
        count = 0
        for p in self.features.parameters():
            if wd > 0 and p.ndim > 1:
                p.mul_(1 - lr * wd)
            u = update[count:count + p.numel()].reshape(p.shape)
            p.add_(u, alpha=-lr)
            count += p.numel()
        for p in self.heads[task_id].parameters():
            if wd > 0 and p.ndim > 1:
                p.mul_(1 - lr * wd)
            u = update[count:count + p.numel()].reshape(p.shape)
            p.add_(u, alpha=-lr)
            count += p.numel()


class ResNet50CL(Module):
    """torchvision ResNet-50 wrapped to satisfy the CL-model interface.

    Same contract as ``ResNet18CL`` (``features`` / ``heads``, GroupNorm by
    default). Used as the larger CLEAR backbone.
    ``feat_dim`` is 2048 (Bottleneck expansion).
    """

    def __init__(self, output_size, num_heads, pretrained=False, norm='gn'):
        super().__init__()
        import torchvision.models as tvm

        if pretrained and norm != 'bn':
            raise ValueError(
                "ResNet-50 pretrained weights are only distributed for BatchNorm; "
                "use norm='bn' together with pretrained=True, or norm='gn' with "
                "pretrained=False."
            )

        norm_layer = _make_norm_layer(norm)
        weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = tvm.resnet50(weights=weights, norm_layer=norm_layer)

        self.features = Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            backbone.avgpool, Flatten(),
        )
        self.feat_dim = backbone.fc.in_features  # 2048 for ResNet-50
        self.heads = ModuleList([
            Linear(self.feat_dim, output_size) for _ in range(num_heads)
        ])

    def forward(self, x, task_id, out='logits'):
        assert out in ('logits', 'features', 'both')
        features = self.features(x)
        if out == 'features':
            return features
        logits = self.heads[task_id](features)
        if out == 'logits':
            return logits
        return logits, features

    def get_gradient(self, task_id):
        g = [p.grad.reshape(-1) for p in self.features.parameters()]
        g += [p.grad.reshape(-1) for p in self.heads[task_id].parameters()]
        return torch.cat(g)

    @torch.no_grad()
    def update(self, update, task_id, lr, wd):
        count = 0
        for p in self.features.parameters():
            if wd > 0 and p.ndim > 1:
                p.mul_(1 - lr * wd)
            u = update[count:count + p.numel()].reshape(p.shape)
            p.add_(u, alpha=-lr)
            count += p.numel()
        for p in self.heads[task_id].parameters():
            if wd > 0 and p.ndim > 1:
                p.mul_(1 - lr * wd)
            u = update[count:count + p.numel()].reshape(p.shape)
            p.add_(u, alpha=-lr)
            count += p.numel()


class ResNetCifar224(Module):
    """Scaled-up ResNet-Cifar sized for 224x224 inputs with a small param budget.

    Unlike the 32x32 ``ResNetCifar``, this variant uses a 7x7 / stride-2 stem
    + 3x3 MaxPool (ImageNet-style) to downsample aggressively at the start,
    keeps 3 stages of ``BasicBlock``s with stride 2 between stages, and ends with
    an AdaptiveAvgPool so the spatial size is resolved automatically.

    At the default ``channels=(32, 64, 128)`` and ``layers=(3, 3, 3)`` this gives
    ~500k-1M parameters, which leaves the DenseMFAC buffer at ~1 GB for m=512 —
    the whole point of this model is to keep Fisher preconditioning tractable on
    raw-image CLEAR.

    Parameters
    ----------
    output_size : int  (number of logits per head)
    num_heads : int    (1 for domain-incremental CLEAR)
    channels : tuple of 3 ints  (stage widths)
    layers : tuple of 3 ints    (blocks per stage)
    norm : {'bn', 'gn'}         (normalization layer, 'gn' recommended for CL)
    """

    def __init__(self, output_size, num_heads, channels=(32, 64, 128), layers=(3, 3, 3), norm='gn'):
        super().__init__()
        assert len(channels) == 3 and len(layers) == 3
        self.output_size = output_size
        self.num_heads = num_heads
        self._norm = _make_norm_layer(norm)

        c0 = channels[0]
        stem = [
            Conv2d(3, c0, kernel_size=7, stride=2, padding=3, bias=False),
            self._norm(c0),
            ReLU(inplace=False),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]

        self._inplanes = c0
        stage1 = self._make_stage(channels[0], layers[0], stride=1)
        stage2 = self._make_stage(channels[1], layers[1], stride=2)
        stage3 = self._make_stage(channels[2], layers[2], stride=2)

        self.features = Sequential(
            *stem, stage1, stage2, stage3,
            AdaptiveAvgPool2d(1),
            Flatten(),
        )
        self.feat_dim = channels[-1]
        self.heads = ModuleList([
            Linear(self.feat_dim, output_size) for _ in range(num_heads)
        ])

        # He init for conv weights (BasicBlock layers already use random init).
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_stage(self, planes, blocks, stride):
        """Build one stage of ``blocks`` BasicBlocks with the first stride=``stride``.

        We reuse the existing ``BasicBlock`` class but supply the GroupNorm-aware
        downsample branch since the original used a hardcoded ``BatchNorm2d``.
        """
        downsample = None
        if stride != 1 or self._inplanes != planes * BasicBlock.expansion:
            downsample = Sequential(
                Conv2d(self._inplanes, planes * BasicBlock.expansion,
                       kernel_size=1, stride=stride, bias=False),
                self._norm(planes * BasicBlock.expansion),
            )

        # block_gates=[True, True] mirrors the ResNetCifar convention
        layers = [BasicBlock([True, True], self._inplanes, planes, stride, downsample)]
        self._inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock([True, True], self._inplanes, planes))

        # Patch in the chosen norm layer on every BasicBlock (the class itself
        # always instantiates BatchNorm2d; swap in GN if requested).
        for blk in layers:
            blk.bn1 = self._norm(blk.bn1.num_features)
            blk.bn2 = self._norm(blk.bn2.num_features)

        return Sequential(*layers)

    def forward(self, x, task_id, out='logits'):
        assert out in ('logits', 'features', 'both')
        features = self.features(x)
        if out == 'features':
            return features
        logits = self.heads[task_id](features)
        if out == 'logits':
            return logits
        return logits, features

    def get_gradient(self, task_id):
        g = [p.grad.reshape(-1) for p in self.features.parameters()]
        g += [p.grad.reshape(-1) for p in self.heads[task_id].parameters()]
        return torch.cat(g)

    @torch.no_grad()
    def update(self, update, task_id, lr, wd):
        count = 0
        for p in self.features.parameters():
            if wd > 0 and p.ndim > 1:
                p.mul_(1 - lr * wd)
            u = update[count:count + p.numel()].reshape(p.shape)
            p.add_(u, alpha=-lr)
            count += p.numel()
        for p in self.heads[task_id].parameters():
            if wd > 0 and p.ndim > 1:
                p.mul_(1 - lr * wd)
            u = update[count:count + p.numel()].reshape(p.shape)
            p.add_(u, alpha=-lr)
            count += p.numel()
