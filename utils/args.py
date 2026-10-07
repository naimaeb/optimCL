import argparse


def _float_or_str_alpha(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def build_parser():
    parser = argparse.ArgumentParser()

    ### WANDB
    parser.add_argument('--wandb_entity', type=str, required=True, help='The wandb entity.')
    parser.add_argument('--wandb_project', type=str, required=True, help='The wandb project inside "entity" owner.')
    parser.add_argument('--wandb_group', type=str, required=True, help='The wandb group in the project.')
    parser.add_argument('--wandb_job_type', type=str, required=True, default=None, help='The wandb job type')
    parser.add_argument('--wandb_name', type=str, required=True, default=None, help='The name for the experiment in wandb runs')

    ### DATASET
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset to use for training (also used as download root for CLEAR).')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name',
                        choices=[
                            'mnist', 'rotmnist', 'rotmnist_gradual', 'rotmnist_cyclic', 'permmnist',
                            'permcontinuous', 'labelpermmnist',
                            'cifar10', 'cifar100', 'rotcifar10',
                            'seqcifar10', 'seqcifar100', 'seqtinyimg',
                            'fivedatasets', 'iblurry',
                            'domainnet', 'core50', 'stream51', 'cladc',
                            'wildtime', 'tic', 'cloc',
                            'clear10', 'clear100',
                            'clear10_neurips2021', 'clear100_cvpr2022',
                            'quadratics', 'bitflip',
                        ])

    ### CLEAR (only used when dataset_name is clear10 / clear100)
    parser.add_argument('--clear_eval_protocol', type=str, required=False, default='streaming',
                        choices=['iid', 'streaming'],
                        help="'iid': 70/30 per-bucket split (requires --clear_seed 0-4); "
                             "'streaming': next-domain evaluation protocol.")
    parser.add_argument('--clear_feature_type', type=str, default= None,
                        choices=['moco_b0', 'moco_imagenet', 'byol_imagenet', 'imagenet', 'None'],
                        help='Pre-extracted feature backbone (all produce 2048-dim vectors).')

    ### MODEL
    parser.add_argument('--model', type=str, required=True,
                        choices=['vit-t', 'vit-s', 'vit-b', 'vit-l',
                                 'logreg', 'ln5', 'mn', 'mlp', 'linreg',
                                 'lenet', 'cstm-lenet',
                                 'rn10t', 'rn', 'rn18', 'rn18cifar', 'rn20', 'rn32', 'rn44', 'rn50', 'rn56',
                                 'rncifar224',
                                 'wrn-22-2', 'wrn-22-4', 'wrn-28-2', 'wrn-28-4', 'wrn-28-8', 'wrn-40-2'],
                        help='Type of model to train.')
    parser.add_argument('--width', type=int, required=False, default=100)

    parser.add_argument('--bias', type=int, default=1, choices=[0, 1])

    # Used only by conv nets trained on raw-image CLEAR.
    parser.add_argument('--pretrained', type=int, default=0, choices=[0, 1],
                        help='If 1, initialize conv-net backbones (rn18, rn50, ...) '
                             'from ImageNet-pretrained weights (requires --bn_type=bn).')
    parser.add_argument('--bn_type', type=str, default='gn', choices=['bn', 'gn'],
                        help='Normalization layer for conv nets. GroupNorm (gn) is '
                             'recommended for continual learning because BatchNorm '
                             'running stats drift across CLEAR buckets and confound '
                             'measurements of the optimizer effect on forgetting.')

    ### TRAINING
    parser.add_argument('--epochs_per_task', type=int, required=False, default=5, help='The number of epochs to train the model for for each task')
    parser.add_argument('--batch_size', type=int, required=True, help='Batchsize to use for training.')
    parser.add_argument('--clip_grad', type=float, default=0,
                        help='Clip global gradient norm if >0; 0 logs norm without clipping.')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['cosine', 'multistep', 'none'],
                        help='Per-task LR schedule. cosine if the dataset file '
                             'defines none (TinyImageNet, CORe50, Rot-MNIST, CLEAR). '
                             'multistep with --lr_milestones for seq-cifar10/100.')
    parser.add_argument('--lr_milestones', type=str, default='35,45',
                        help='Comma-separated epoch milestones for multistep LR '
                             '(Mammoth seq-cifar100: 35,45).')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Multiplicative decay at each multistep milestone.')

    parser.add_argument('--fisher_capacity', type=int, required=False, default=64)
    parser.add_argument('--fisher_damp', type=float, required=False, default=1e-5)
    parser.add_argument('--fisher_density', type=float, required=False, default=0.01)
    # Recency bias for the M-FAC Fisher refill (and the matching ER replay schedule).
    # 0 = uniform (current behaviour): every seen task contributes ~capacity/(t+1) batches.
    # 1 = geometric: task j gets weight `recency_decay**j`, then the buffer capacity is
    #     split proportionally with a floor of 1 batch/task. Higher decay = stronger
    #     emphasis on recent tasks. decay=1.0 is exactly uniform.
    parser.add_argument('--mfac_recency_bias', type=int, default=0, choices=[0, 1],
                        help='Recency bias for the M-FAC Fisher refill schedule. '
                             '0 = uniform per-task budget (default). '
                             '1 = geometric weights so newer tasks contribute more '
                             'gradients per Fisher refill.')
    parser.add_argument('--mfac_recency_decay', type=float, default=2.0,
                        help='Geometric decay base used when --mfac_recency_bias=1. '
                             'w_j = decay**j with j=0 (oldest seen task) ... '
                             'j=t (current task). decay=1.0 reproduces uniform; '
                             'decay>1 increases the share allocated to recent tasks.')

    parser.add_argument('--optim', required=True, help='Type of optimizer to use for training.', choices=[
        'ademamix', 'replaysgd',
    ])
    parser.add_argument('--weight_decay', type=float, default=0, required=False, help='Weight decay to use for the optimizer.')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    ### ADAM / ADAMW / (BLOCK) ADAM
    parser.add_argument('--beta1', type=float, required=False, default=0.9,
                        help='Fast/normal first-moment coefficient. Muon momentum, '
                             'and the fast EMA of AdEMAMix, Replay, and Double optimizers.')
    parser.add_argument('--beta2', type=float, required=False, default=0.999,
                        help='Second-moment coefficient (Adam β2). Not a momentum on SGD/Muon.')
    #PARAMETERS FOR ADEMAMIX
    parser.add_argument('--beta3', type=float, required=False, default=0.9999,
                        help='Slow first-moment coefficient for AdEMAMix, Replay, and Double optimizers. '
                             'Overwritten when --horizon_ratio is set: β3 = rL/(1+rL).')
    parser.add_argument('--horizon_ratio', type=float, required=False, default=None,
                        help='Slow-memory horizon in tasks, r = τ/L. '
                             'L = ceil(N_task/batch)*epochs from DATASET_DETAILS '
                             '(median over loaded tasks when the dataset is unlisted). '
                             'τ = r*L and β3 = τ/(1+τ). The same β is written to β4 when '
                             '--beta4_from_horizon 1, or when a positive --beta4 is passed '
                             'alongside this flag. None leaves --beta3 unchanged. '
                             'Values with r > tasks/4 are transient (EMA not 98%% full).')
    parser.add_argument('--beta4_from_horizon', type=int, required=False, default=None,
                        choices=[0, 1],
                        help='With --horizon_ratio: 1 sets β4 = β3 = rL/(1+rL) so the slow '
                             'second moment reaches the same past gradients; 0 sets β4 = 0 '
                             '(current gradient only). None leaves --beta4 unchanged, except '
                             'that a positive --beta4 is still replaced by rL/(1+rL).')
    parser.add_argument('--alpha', type=_float_or_str_alpha, required=False, default=2.0,
                        help='AdEMAMix numeric alpha, or double-optimizer mix schedule '
                             'avg / task / inv-task (see double_mix_alpha).')
    parser.add_argument('--beta3_warmup', type=int, required=False, default=10_000,
                        help='Number of warmup steps to increase beta3 (β1→β3). '
                             'Overridden at runtime when --warmup_tasks > 0.')
    parser.add_argument('--alpha_warmup', type=int, required=False, default=10_000,
                        help='Number of warmup steps to increase alpha (0→α). '
                             'Overridden at runtime when --warmup_tasks > 0.')
    parser.add_argument('--warmup_tasks', type=float, required=False, default=0.0,
                        help='If > 0, set beta3_warmup = alpha_warmup = round(warmup_tasks * L) '
                             'where L is the per-task optimizer step count. Warmup counts global '
                             'optimizer steps, so warmup_tasks=1 warms the slow term over task 0. '
                             'Supported by ademamix, replaysgd, and replaymuon.')
    parser.add_argument('--beta4', type=float, required=False, default=None,
                        help='Slow second-moment decay rate for AdEMAMix. '
                             'When set, a separate exp_avg_sq_slow buffer is maintained '
                             'and each EMA branch uses its own denominator '
                             '(fast branch: beta2, slow branch: beta4). '
                             'With --horizon_ratio, a positive value is replaced by '
                             'β4 = rL/(1+rL) (same past-gradient reach as β3). '
                             '0 keeps the buffer but gives it no history. '
                             'None (default) disables the slow second moment and '
                             'restores original AdEMAMix behaviour.')
    # Boundary-free primal anchor. Diagonal-only; no gating/KFAC.
    parser.add_argument('--anchor_mu', type=float, required=False, default=None,
                        help='Primal-anchor weight for AdEMAMix (Eq. "eq:anchor"). '
                             'When set, replaces the plain gradient step with a proximal '
                             'update that also pulls toward a slow EMA of the iterate '
                             '(theta_bar) under the historical-Fisher metric diag(v_t^s). '
                             'Requires --beta4 to be set. None (default) disables the '
                             'anchor and restores the plain AdEMAMix update.')
    parser.add_argument('--anchor_tau', type=float, required=False, default=None,
                        help='EMA retention rate for the primal anchor point theta_bar '
                             '(theta_bar_t = anchor_tau*theta_bar_{t-1} + (1-anchor_tau)*'
                             'theta_{t-1}). Only used when --anchor_mu is set. '
                             'Typical value: matched to beta3\'s half-life.')
    # Dual/metric-channel ablation flag. Overrides --alpha/--beta4 right after
    # arg parsing.
    parser.add_argument('--channel_variant', type=str, required=False, default=None,
                        choices=['dual_only', 'metric_only', 'both'],
                        help='[deprecated for naive.py: set --alpha / --beta4 directly] AdEMAMix dual-memory (alpha/exp_avg_slow) vs '
                             'metric-memory (beta4/exp_avg_sq_slow) channel ablation. '
                             '"dual_only": forces --beta4 None (keeps --alpha as given). '
                             '"metric_only": forces --alpha 0 (keeps --beta4 as given). '
                             '"both": no-op, uses --alpha/--beta4 as given. '
                             'None (default): no override.')

    # Per-layer momentum / squared-gradient / preconditioner diagnostics.
    # 1 (default) wires `OptimGeometryTracker` between loss.backward() and
    # optimizer.step() in the supported training loops (currently `naive.py`)
    # and logs per-epoch means under `optim_geom/*` in W&B. 0 disables the
    # extra forward/backward-free work (small but non-zero per step).
    parser.add_argument('--track_optim_geom', type=int, default=1, choices=[0, 1],
                        help='Log per-layer cos(m_{t-1}, g_t), ||m||/||g||, '
                             'cos(v_{t-1}, g_t^2), ||g^2||/||v|| and '
                             '||g/(sqrt(v)+eps)||/||g|| each epoch. '
                             'Reads momentum_buffer / exp_avg / exp_avg_fast / '
                             'exp_avg_slow / exp_avg_sq from optimizer state.')
    parser.add_argument('--optim_geom_every', type=int, default=10,
                        help='Log `optim_geom/*` every N epochs (per task) '
                             'instead of every epoch, to cut per-step recording '
                             'overhead and W&B history volume. The first and '
                             'last epoch of each task are always logged so task '
                             'boundaries are captured. Only relevant when '
                             '--track_optim_geom 1.')
    # Geometric-drift tracker (GeometricDriftTracker in utils/diagnostics.py).
    # Only meaningful when --optim ademamix, since it reads exp_avg_slow from
    # the optimizer state.  When enabled, logs `drift/cosine_sim` and
    # `drift/relative_l2` every --drift_eval_every epochs during the new task,
    # measuring how far the current gradient (on the half-life replay window)
    # has drifted from the slow-momentum snapshot captured at the task boundary.
    parser.add_argument('--track_drift', type=int, default=1, choices=[0, 1],
                        help='Enable GeometricDriftTracker (AdEMAMix only). '
                             'Logs drift/cosine_sim and drift/relative_l2 '
                             'to W&B during the new task. '
                             'Auto-disabled with a warning for non-AdEMAMix optimizers. '
                             '1 = enabled (default), 0 = disabled.')
    parser.add_argument('--drift_eval_every', type=int, default=1,
                        help='Evaluate drift metrics every N training epochs '
                             'of the new task (default 1 = every epoch). '
                             'Increase to reduce the cost of the extra backward '
                             'pass on the half-life buffer.')
    # Reference (past-parameter) gradient drift. When enabled, the tracker also
    # snapshots g_ref = grad(buffer, theta_T) at the task boundary -- a real
    # backward pass on the half-life buffer at the end-of-task weights -- and
    # logs drift/{cosine_sim,log_scale_ratio}{,_sq}_ref comparing the current
    # gradient against that fixed-theta_T reference. This isolates pure
    # parameter drift (same data, same fixed past weights) but stores an extra
    # full-size gradient vector per task boundary, hence the opt-in flag.
    parser.add_argument('--track_drift_ref', type=int, default=0, choices=[0, 1],
                        help='Also log reference (past-parameter) drift metrics '
                             'drift/cosine_sim_ref, drift/log_scale_ratio_ref '
                             '(and _sq variants), comparing the current gradient '
                             'on the buffer against grad(buffer, theta_T) frozen '
                             'at the task boundary. Extra memory: one gradient '
                             'vector. 0 = disabled (default), 1 = enabled.')

    # Gradient-noise proxy. See utils/gradient_noise.py -- estimates
    # sigma^2 = mean_i ||g_i - g_bar||^2 via torch.func.vmap(torch.func.grad(...))
    # on a single minibatch.
    parser.add_argument('--track_grad_noise', type=int, default=0, choices=[0, 1],
                        help='Estimate per-example gradient-noise variance sigma^2 '
                             '(utils.gradient_noise.estimate_gradient_noise) every '
                             '--grad_noise_every_epochs epochs of the *first* task '
                             'and log it as grad_noise/sigma_sq. Extra cost: one '
                             'vmapped forward+backward on a single minibatch per '
                             'measurement (not every step). 0 = disabled (default).')
    parser.add_argument('--grad_noise_every_epochs', type=int, default=1,
                        help='Estimate grad_noise/sigma_sq every N epochs of the '
                             'first task (default: every epoch of task 0 only -- '
                             'sigma^2 is a property of the data/batch-size, not '
                             'the optimizer or task index, so one task is enough). '
                             'Only relevant when --track_grad_noise 1.')

    parser.add_argument('--blockadam_block_size', type=int, required=False, default=32)
    parser.add_argument('--blockadam_ev_func', type=str, required=False, default='ns', choices=['abs', 'relu', 'ns'])
    parser.add_argument('--blockadam_use_sqrt', type=int, required=False, default=1, choices=[0, 1])
    parser.add_argument('--nesterov', type=int, default=1, choices=[0, 1],
                        help='Use Nesterov-style lookahead in SGD/Muon double optimizers.')
    parser.add_argument('--version', type=int, default=1, choices=[1, 2],
                        help='DoubleMuon update version.')

    ### CONTINUAL LEARNING
    parser.add_argument('--use_multihead_model', type=int, required=False, default=1, choices=[0, 1],
                        help='[ignored by naive.py] Forced to 0 by apply_il_protocol (single Linear). '
                             'Task-IL is masked CE, not separate head modules.')
    parser.add_argument('--tasks', type=int, required=True, help='The number of tasks to train on')
    parser.add_argument('--angle', type=int, default=0,
                        help='Rotation step in degrees between consecutive rotmnist '
                             'tasks, the per-task rotation span for rotmnist_gradual, '
                             'and the legacy rotmnist_cyclic sawtooth spacing.')
    parser.add_argument('--cycle_period', type=int, default=0,
                        help='[deprecated for naive.py: use --stream_type / --cycle_slots] Legacy rotmnist_cyclic sawtooth: task t uses '
                             '(t mod P) * --angle. Ignored when --stream_type is set.')
    parser.add_argument('--stream_type', type=str, default='',
                        choices=['', 'cyclic', 'onepass'],
                        help='rotmnist_cyclic schedule. cyclic ping-pongs '
                             '0,δ,...,hδ,...,δ with h = cycle_slots // 2; '
                             'onepass is the monotone sweep k*δ. '
                             'Empty keeps the legacy cycle_period sawtooth.')
    parser.add_argument('--angle_delta', type=int, default=0,
                        help='Degrees between consecutive rotmnist_cyclic tasks. '
                             'Both cyclic and onepass use this same |Δ|. '
                             '0 falls back to --angle.')
    parser.add_argument('--cycle_slots', type=int, default=6,
                        help='Tasks per ping-pong cycle (P). '
                             '--tasks must be a multiple of P when '
                             '--stream_type cyclic. Also sets the logged '
                             'cycle length C = P * L on both streams.')
    parser.add_argument('--perm_frac', type=float, default=0.0,
                        help='Fraction of pixels permuted on permmnist / permcontinuous.')
    parser.add_argument('--label_perm_frac', type=float, default=1.0,
                        help='Fraction of the label map reassigned on labelpermmnist. '
                             'Task 0 stays identity.')
    parser.add_argument('--classes_per_task', type=int, default=2,
                        help='Disjoint classes per split-CIFAR / TinyImageNet task.')
    parser.add_argument('--task_order', type=str, default='',
                        help='Comma-separated 5-Datasets order, e.g. '
                             'mnist,svhn,notmnist,fashionmnist,cifar10.')
    parser.add_argument('--iblurry_n', type=float, default=50.0,
                        help='i-Blurry disjoint-class percentage (100 = split CIFAR, '
                             '0 = fully shared classes).')
    parser.add_argument('--transition_width', type=float, default=1.0,
                        help='Gaussian width σ for permcontinuous. σ→0 recovers hard '
                             'permmnist; larger σ = softer task transitions.')
    parser.add_argument('--domain_list', type=str, default='',
                        help='Comma-separated domain/session folder names under '
                             'dataset_path (DomainNet, CORe50, CLAD, Wild-Time, TiC, CLOC). '
                             'Empty = sorted subdirectories.')
    parser.add_argument('--image_size', type=int, default=0,
                        help='Resize folder-stream images to this size. 0 = dataset default.')
    parser.add_argument('--bitflip_n_bits', type=int, default=4,
                        help='How many coordinates of w* flip sign at each bitflip task.')
    # Experience replay (data memory) baseline, for comparison with optimizer
    # memory (AdEMAMix beta3 / beta4) in the same naive.py loop.
    parser.add_argument('--replay', type=int, default=0, choices=[0, 1],
                        help='If 1, keep a reservoir of past-task examples and '
                             'mix one replay minibatch into each current-task '
                             'step (weighted CE/MSE). Implied by --ablation replay. '
                             '0 = disabled (default).')
    parser.add_argument('--replay_buffer_size', type=int, default=1000,
                        help='Experience-replay capacity in examples. '
                             'Not the optimizer FIFO/reservoir (--buffer_size), '
                             'which counts minibatches. Used when --replay 1. '
                             'Overridden when --replay_per_angle > 0.')
    parser.add_argument('--replay_per_angle', type=int, default=0,
                        help='Scaled ER capacity: if > 0, K = m * n_distinct_angles '
                             'on rotmnist_cyclic (overrides --replay_buffer_size). '
                             'Holds exemplars per angle fixed as T grows.')
    parser.add_argument('--ablation', type=str, default=None,
                        choices=[
                            'slow_ema',
                            'slow_ema_refresh',
                            'slow_ema_reset',
                            'fifo_geometric_fresh',
                            'reservoir_geometric_stale',
                            'fifo_uniform_stale',
                            'reservoir_uniform_stale',
                            'reservoir_uniform_fresh',
                            'fast_only',
                            'replay',
                        ],
                        help='ReplaySGD memory ablation. slow_ema and the four buffered '
                             'estimators use the bias-corrected mix and keep mu at 0 on '
                             'task 0. fast_only is that mix with mu left at 0. replay '
                             'keeps the legacy step and mixes a reservoir minibatch '
                             'into the loss.')
    parser.add_argument('--buffer_size', type=int, default=500,
                        help='Optimizer-memory reservoir capacity in minibatches '
                             '(one stored gradient or index row per step). '
                             'Not --replay_buffer_size, which counts examples. '
                             'Used by reservoir_geometric_stale and reservoir_uniform_stale. '
                             'FIFO windows ignore this and use ceil(2*tau) batches.')
    parser.add_argument('--memory_examples', type=int, default=0,
                        help='Example budget K of the fresh (recomputed) memories, '
                             'counted like --replay_buffer_size. fifo_geometric_fresh keeps '
                             'each example with probability K/(M*B), so a full window '
                             'holds K; 0 keeps the whole window. Required by '
                             'reservoir_uniform_fresh (Vitter reservoir of K examples).')
    parser.add_argument('--memory_samples', type=int, default=0,
                        help='How many individual examples are drawn from the '
                             'stored batches and averaged into mu. 0 uses '
                             'batch_size. Not a count of stored batches.')
    parser.add_argument('--memory_dtype', type=str, default='bf16', choices=['bf16', 'fp32'],
                        help='Storage dtype for stale minibatch gradients.')

    ### STALENESS PROBE AND REFRESHED SLOW-EMA ARM
    parser.add_argument('--probe_staleness', type=int, default=0, choices=[0, 1],
                        help='Run the slow-EMA staleness projection probe. Requires '
                             '--ablation slow_ema.')
    parser.add_argument('--probe_points', type=str, default='boundary,0.25,0.5,end',
                        help='Comma-separated fractions of task length L at which the probe '
                             'runs ("boundary"=0.0, "end"=1.0). Only task_id>=1.')
    parser.add_argument('--probe_chunk', type=int, default=64,
                        help='Number of past batches recomputed per vmap chunk inside the probe.')
    parser.add_argument('--probe_heldout', type=int, default=2000,
                        help='Held-out examples per past task for the probe reference gradient '
                             'G_old = grad(loss on H; theta_n).')
    parser.add_argument('--probe_sigma', type=int, default=0, choices=[0, 1],
                        help='Also run the second probe pass that computes sigma_s = w_s '
                             '<Delta_s, E> / ||E||^2 per past batch. Doubles probe cost.')
    parser.add_argument('--probe_window', type=int, default=0,
                        help='If >0, truncate the probe sum to the last k*tau steps.')
    parser.add_argument('--probe_stride', type=int, default=1,
                        help='Subsample stride for per-batch rows in the probe output '
                             '(the aggregate row stays exact).')
    parser.add_argument('--steplog_dtype', type=str, default='fp32', choices=['fp32', 'bf16'],
                        help='Storage dtype for the StepLog G column (per-step clipped '
                             'gradient history, shared by probe and slow_ema_refresh).')
    parser.add_argument('--steplog_store_theta', type=int, default=0, choices=[0, 1],
                        help='Also store the full-model fp32 parameter vector θ_s at '
                             'every logged step (StepLog memory ×2). Needed by the §1 '
                             'linear/NL split and by the §2 quadratic-model probe '
                             'validation driver.')
    parser.add_argument('--steplog_store_grad', type=int, default=1, choices=[0, 1],
                        help='Store the per-step gradient G_s in the StepLog (default). '
                             'Set to 0 for ResNet-scale models: slow_ema_refresh then '
                             'drops the control-variate term in its correction '
                             '(C = mean_k iw_k·g_new_k − μ_stale instead of '
                             'mean_k iw_k·(g_new_k − G_s_k)); still unbiased, higher '
                             'variance. Required for the per-batch staleness probe.')
    parser.add_argument('--refresh_batches', type=int, default=4,
                        help="S' for slow_ema_refresh: number of past batches sampled per step "
                             'to form the correction C. 0 collapses the arm to slow_ema.')
    parser.add_argument('--refresh_sampling', type=str, default='systematic',
                        choices=['systematic', 'iid'],
                        help='Sampling scheme for the refresh correction draws.')
    parser.add_argument('--refresh_importance', type=str, default='none',
                        choices=['none', 'age'],
                        help='If "age", draw with q_s ∝ w_s · (1 + age_s/tau) and weight each '
                             'term by w_{s_k}/q_{s_k}. Unbiased.')
    parser.add_argument('--rho_sam', type=float, default=0.0,
                        help='SAM perturbation radius. 0 disables SAM.')
    parser.add_argument('--sam_mode', type=str, default='none',
                        choices=['none', 'task0_only', 'all'],
                        help='When to apply SAM. task0_only: only during task 0 '
                             '(stability-at-start experiment). all: every step.')

    ### TASK-BOUNDARY + MID-TASK STALENESS PROBE
    # Independent of --probe_staleness: this probe snapshots θ at task
    # boundaries (and the task midpoint) and compares gradients on a
    # held-out past-task batch at θ_now vs θ_task_end / θ_task_mid. Measures
    # slow-EMA-weighted past-task staleness without the full StepLog that
    # stale_probe requires. CIFAR-safe (~0.5 GB CPU per run, ≤1% overhead).
    parser.add_argument('--task_boundary_probe', type=int, default=0, choices=[0, 1],
                        help='Enable the task-boundary + mid-task staleness probe. '
                             'Writes probe/task_boundary_{aggregate,perpasttask}.csv '
                             'under --output_folder. Requires --ablation slow_ema.')
    parser.add_argument('--probe_batch_size', type=int, default=128,
                        help='Held-out probe batch size per past task.')
    parser.add_argument('--probe_task_fractions', type=str, default='0.25,0.5,0.75,1.0',
                        help='Comma-separated task-fractions at which the task-boundary '
                             'probe fires (within every task t >= 1). Fraction 1.0 is '
                             'the last batch of the task.')
    parser.add_argument('--probe_snapshot_dtype', type=str, default='bf16',
                        choices=['bf16', 'fp32'],
                        help='Storage dtype for the task-end and task-mid θ snapshots.')

    ### QUADRATICS (only used when dataset_name is quadratics; see utils/quadratic_tasks.py)
    parser.add_argument('--quad_dim', type=int, default=100,
                        help='Dimension d of the shared weight / feature space.')
    parser.add_argument('--quad_n_train', type=int, default=1000,
                        help='Training samples per quadratic task.')
    parser.add_argument('--quad_n_test', type=int, default=200,
                        help='Test samples per quadratic task.')
    parser.add_argument('--quad_eigval_min', type=float, default=0.1,
                        help='Lower bound for log-uniform Hessian eigenvalue draws.')
    parser.add_argument('--quad_eigval_max', type=float, default=10.0,
                        help='Upper bound for log-uniform Hessian eigenvalue draws.')
    parser.add_argument('--quad_w_star_scale', type=float, default=3.0,
                        help='L2 norm of each task optimum ||w*_t||; larger -> more forgetting.')

    parser.add_argument('--min_train_loss', type=float, default=0, help='The minimum trainin loss to reach in each task. Set to 0 to disable.')

    ### I/O
    parser.add_argument('--output_folder', type=str, required=True, help='Name of the file where the checkpoint of the most recent epoch is persisted.')

    ### OTHERS
    parser.add_argument('--seed', type=int, required=True, help='The seed used to initialize the random number generator')

    ### initialization experiment: train a model with SGD and Shampoo for a single task to a specific training loss and save the checkpoint.
    ### After that, continue with SGD and/or SHMP to check the performance
    parser.add_argument('--init_exp_enable', type=int, required=False, default=0, choices=[0, 1])
    parser.add_argument('--init_exp_init_opt', type=str, required=False, default=0, choices=['sgd', 'shmp'])
    parser.add_argument('--init_exp_init_lr', type=float, required=False, default=1e-3)

    return parser


def get_arg_parse(argv=None):
    return build_parser().parse_args(argv)
