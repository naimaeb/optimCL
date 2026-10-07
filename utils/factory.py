"""Model and optimizer construction from parsed arguments."""
from utils.il_protocol import NUM_CLASSES
from utils.ladder_datasets import FOLDER_SPECS, discover_domains, shared_class_to_idx
from utils.models import *
from optimizers import (
    AdEMAMix,
    ReplaySGD,
)


def get_model(args): # model_name, dataset_name, args, pretrained=False):
    model_name = args.model
    dataset_name = args.dataset_name
    num_classes = NUM_CLASSES[dataset_name]
    if dataset_name in FOLDER_SPECS:
        try:
            num_classes = len(shared_class_to_idx(discover_domains(args)))
            print(f'{dataset_name}: inferred n_classes={num_classes} from domain folders')
        except Exception as exc:
            print(f'WARNING: could not scan {dataset_name} folders ({exc}); '
                  f'using NUM_CLASSES={num_classes}')

    if model_name == 'mlp':
        if any(k in dataset_name for k in ('mnist', 'permcontinuous')):
            input_size = 28 * 28
        elif 'cifar' in dataset_name or dataset_name in ('fivedatasets', 'iblurry'):
            input_size = 3 * 32 * 32
        elif 'clear' in dataset_name:
            input_size = 2048
        else:
            raise RuntimeError(f'The MLP model is not supported for the {dataset_name} yet, please use MNIST or CIFAR!')
        return MultiHeadMLP(
            input_size=input_size,
            hidden_size=args.width,
            output_size=num_classes,
            num_heads=args.tasks if (args.use_multihead_model and 'clear' not in dataset_name) else 1,
            bias=bool(args.bias))
    if 'lenet' in model_name:
        if 'mnist' in dataset_name:
            return LeNet5_MNIST(output_size=num_classes, num_heads=args.tasks if args.use_multihead_model else 1, hidden_size=args.width if model_name == 'cstm-lenet' else None)
        if 'cifar' in dataset_name:
            return LeNet5_CIFAR(output_size=num_classes, num_heads=args.tasks if args.use_multihead_model else 1, hidden_size=args.width if model_name == 'cstm-lenet' else None)
    if model_name == 'rn20':
        return resnet20(output_size=num_classes, num_heads=args.tasks if args.use_multihead_model else 1)
    if model_name == 'rn18cifar':
        return ResNet18CifarCL(
            output_size=num_classes,
            num_heads=args.tasks if args.use_multihead_model else 1,
            nf=64,
            norm=getattr(args, 'bn_type', 'bn') or 'bn',
        )

    # Conv nets: torchvision ResNet-18/50 (any spatial size via adaptive pool).
    # Domain-IL streams (CLEAR, CORe50, DomainNet, ...) use a single head.
    _domain_il = (
        'clear' in dataset_name
        or dataset_name in (
            'domainnet', 'core50', 'stream51', 'cladc',
            'wildtime', 'tic', 'cloc',
            'rotmnist', 'rotmnist_gradual', 'rotmnist_cyclic', 'permmnist', 'permcontinuous',
            'labelpermmnist',
        )
    )
    n_heads = 1 if _domain_il else (args.tasks if args.use_multihead_model else 1)
    if model_name == 'rn18':
        return ResNet18CL(
            output_size=num_classes,
            num_heads=n_heads,
            pretrained=bool(args.pretrained),
            norm=args.bn_type)
    if model_name == 'rn50':
        return ResNet50CL(
            output_size=num_classes,
            num_heads=n_heads,
            pretrained=bool(args.pretrained),
            norm=args.bn_type)
    if model_name == 'rncifar224':
        return ResNetCifar224(
            output_size=num_classes,
            num_heads=n_heads,
            norm=args.bn_type)
    if model_name == 'linreg':
        if dataset_name not in ('quadratics', 'bitflip'):
            raise RuntimeError("linreg model is only supported for quadratics/bitflip.")
        return LinearRegressorCL(input_size=getattr(args, 'quad_dim', 100))
    raise RuntimeError(f'Model {model_name} is not implemented!')


def get_param_groups(model, weight_decay):
    return param_groups_weight_decay(model=model, weight_decay=weight_decay)


def double_mix_alpha(alpha, task_id):
    """Mix weight for DoubleAdamW / DoubleMuon / DoubleSGD.

    Numeric ``alpha`` is used as-is. Named schedules:
      avg      -> 0.5
      task     -> 1 on task 0, else 1/(t+1)
      inv-task -> 1 on task 0, else 1 - 1/(t+1)
    """
    if isinstance(alpha, str):
        if alpha == 'avg':
            return 0.5
        if alpha == 'task':
            return 1.0 if task_id == 0 else 1.0 / (task_id + 1)
        if alpha == 'inv-task':
            return 1.0 if task_id == 0 else 1.0 - 1.0 / (task_id + 1)
        raise ValueError(
            "alpha must be a float or one of 'avg', 'task', 'inv-task'; "
            f"got {alpha!r}")
    return float(alpha)


def get_optimizer(args, model_params, model_modules):
    optimizer = None
    if args.optim == 'ademamix':
        optimizer = AdEMAMix(
            model_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2, args.beta3),
            alpha=args.alpha,
            beta3_warmup=args.beta3_warmup,
            alpha_warmup=args.alpha_warmup,
            beta4 = getattr(args, 'beta4', None),
            weight_decay=args.weight_decay,
            anchor_mu=getattr(args, 'anchor_mu', None),
            anchor_tau=getattr(args, 'anchor_tau', None),
            ablation=getattr(args, 'ablation', None),
            tau=float(getattr(args, 'cfg_tau', 0.0) or 0.0),
            batch_size=int(args.batch_size),
            total_steps=int(getattr(args, 'cfg_total_steps', 0) or 0),
            steplog_dtype=getattr(args, 'steplog_dtype', 'fp32') or 'fp32',
            steplog_store_theta=bool(int(getattr(args, 'steplog_store_theta', 0) or 0)),
            steplog_store_grad=bool(int(getattr(args, 'steplog_store_grad', 1) or 0)),
            refresh_batches=int(getattr(args, 'refresh_batches', 0) or 0),
            refresh_sampling=getattr(args, 'refresh_sampling', 'systematic') or 'systematic',
            refresh_importance=getattr(args, 'refresh_importance', 'none') or 'none',
            seed=int(args.seed))

    elif args.optim == 'replaysgd':
        assert isinstance(args.alpha, float)
        optimizer = ReplaySGD(
            model_params, lr=args.lr, weight_decay=args.weight_decay, beta1=args.beta1,
            beta3=args.beta3, alpha=args.alpha, nesterov=bool(args.nesterov),
            beta3_warmup=args.beta3_warmup, alpha_warmup=args.alpha_warmup,
            ablation=getattr(args, 'ablation', None),
            tau=float(getattr(args, 'cfg_tau', 0.0) or 0.0),
            buffer_size=int(getattr(args, 'buffer_size', 500) or 500),
            batch_size=int(args.batch_size),
            memory_samples=int(getattr(args, 'memory_samples', 0) or 0),
            memory_dtype=getattr(args, 'memory_dtype', 'bf16') or 'bf16',
            seed=int(args.seed),
            memory_examples=int(getattr(args, 'memory_examples', 0) or 0),
            total_steps=int(getattr(args, 'cfg_total_steps', 0) or 0),
            steplog_dtype=getattr(args, 'steplog_dtype', 'fp32') or 'fp32',
            steplog_store_theta=bool(int(getattr(args, 'steplog_store_theta', 0) or 0)),
            steplog_store_grad=bool(int(getattr(args, 'steplog_store_grad', 1) or 0)),
            probe_staleness=bool(int(getattr(args, 'probe_staleness', 0) or 0)),
            refresh_batches=int(getattr(args, 'refresh_batches', 0) or 0),
            refresh_sampling=getattr(args, 'refresh_sampling', 'systematic') or 'systematic',
            refresh_importance=getattr(args, 'refresh_importance', 'none') or 'none')

    if optimizer is None:
        raise RuntimeError(f'Optimizer {args.optim} is not implemented!')

    if isinstance(optimizer, list):
        for opt in optimizer:
            opt.zero_grad(set_to_none=True)
    else:
        optimizer.zero_grad(set_to_none=True)
    return optimizer
