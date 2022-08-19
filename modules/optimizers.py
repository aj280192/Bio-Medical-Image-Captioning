import torch


def r2g_optimizer(args, model):
    ve_params = list(map(id, model.visual_extractor.parameters()))
    ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
    optimizer = getattr(torch.optim, args.optim)(
        [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
         {'params': ed_params, 'lr': args.lr_ed}],
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer

def sat_optimizer(args, model):
    ve_params = filter(lambda p: p.requires_grad, model.visual_extractor.parameters())
    ed_params = filter(lambda p: p.requires_grad, model.encoder_decoder.parameters())
    optimizer = getattr(torch.optim, args.optim)(
        [{'params': ve_params, 'lr': args.lr_ve},
         {'params': ed_params, 'lr': args.lr_ed}],
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer

def normal_optimizer(args, model):
    optimizer = getattr(torch.optim, args.optim)(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer


def s2s_optimizer(args, model):
    optimizer = getattr(torch.optim, args.optim)(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def build_lr_scheduler(args, optimizer):

    # if args.model_type == 'R2G':
    #     lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    #
    # elif args.model_type == 'SAT':
    #     lr_scheduler = getattr(torch.optim.lr_scheduler, 'ReduceLROnPlateau')(optimizer, mode='max', factor=0.8, patience=8)
    #
    # else:
    #     lr_scheduler = getattr(torch.optim.lr_scheduler, 'LambdaLR')(optimizer, lr_lambda=lambda step: rate(
    #         step, model_size=args.emb_dim, factor=1.0, warmup=1))

    # if args.model_type != 'SAT':
    #     lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    #
    # else:
    #     lr_scheduler = getattr(torch.optim.lr_scheduler, 'ReduceLROnPlateau')(optimizer, mode='max', factor=0.8, patience=8)

    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)

    return lr_scheduler


