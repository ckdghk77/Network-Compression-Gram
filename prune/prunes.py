import torch
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d as CV2D
from torch.nn.modules.batchnorm import BatchNorm2d as BN2D
from torch.nn.modules.linear import Linear as LN

import torch_pruning as tp
from functools import partial

def _explore_all_layers(net, n_f_dict, prefix=""):
    for name, layer in net._modules.items():
        # If it is a sequential, don't register a hook on it
        # but recursively register hook on all it's module children
        if len(layer._modules):
            _explore_all_layers(layer, n_f_dict, '{}_'.format(prefix + name))
        else:
            # it's a non sequential. Register a hook
            n_f_dict[prefix+name] = layer

def _ignore_params(net, _layers, kmethod) :
    for name, layer in net._modules.items():
        if len(layer._modules):
            _ignore_params(layer, _layers, kmethod)
        else:
            if hasattr(layer, 'weight') and type(layer) == LN :
                _layers.append(layer)
def train_model(
        model,
        train_loader,
        epochs,
        lr,
        lr_decay_milestones,
        lr_decay_gamma=0.1,
        # For pruning
        weight_decay=5e-4,
        pruner=None,
        device=None,
):

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay if pruner is None else 0,
    )
    milestones = [int(ms) for ms in lr_decay_milestones.split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=lr_decay_gamma
    )

    for epoch in range(epochs):
        model.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target)
            loss.backward()
            pruner.regularize(model)  # for sparsity learning
            optimizer.step()

            if i % 10 == 0 :
                print(
                    "Epoch {:d}/{:d}, iter {:d}/{:d}, loss={:.4f}, lr={:.4f}".format(
                        epoch,
                        epochs,
                        i,
                        len(train_loader),
                        loss.item(),
                        optimizer.param_groups[0]["lr"],
                    )
                )

        if isinstance(pruner, tp.pruner.GrowingRegPruner):
            pruner.update_reg()  # increase the strength of regularization
            # print(pruner.group_reg[pruner._groups[0]])

        scheduler.step()


def global_prunes(model, pmethod, kmethod, pamount, speed_up, train_loader, gpu):
    _ign_layers = []
    n_f_dict = dict();
    # Explore all layers and name it
    _explore_all_layers(model, n_f_dict=n_f_dict, prefix="");
    # Ignored Layers
    _ign_layers = [n_f_dict['fc']]
    if kmethod == 1 :
        _ign_layers.extend([n_f_dict['bn1'], n_f_dict['layer1_2_bn3'],
                            n_f_dict['layer2_3_bn3'], n_f_dict['layer3_5_bn3'],
                            n_f_dict['layer4_2_bn3']])
    elif kmethod == 2 :
        _ign_layers.extend([n_f_dict['relu'], n_f_dict['layer1_2_relu'],
                            n_f_dict['layer2_3_relu'], n_f_dict['layer3_5_relu'],
                            n_f_dict['layer4_2_relu']])
    elif kmethod == 3 :
        _ign_layers.extend([n_f_dict['layer1_1_conv1'], n_f_dict['layer1_2_bn3'],
                           n_f_dict['layer2_1_conv1'], n_f_dict['layer2_3_bn3'],
                           n_f_dict['layer3_1_conv1'], n_f_dict['layer3_5_bn3'],
                           n_f_dict['layer4_1_conv1'], n_f_dict['layer4_2_bn3']])

    # 1. Importance criterion
    if pmethod == "BN" :
        imp = tp.importance.BNScaleImportance()  # or GroupTaylorImportance(), GroupHessianImportance(), etc.
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=1e-5, global_pruning=True,
                               group_lasso=True)
    elif pmethod == "L1" :
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=True)

    elif pmethod == "GN" :
        imp = tp.importance.GroupNormImportance(p=2, normalizer='max')
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=1e-5, global_pruning=True)

    example_inputs = torch.rand(size=(1,3,32,32));

    pruner = pruner_entry(  # We can always choose MetaPruner if sparse training is not required.
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=pamount,  # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
        ignored_layers=_ign_layers,
        round_to=8,
        # It's recommended to round dims/channels to 4x or 8x for acceleration. Please see: https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
    )

    if pmethod in ["BN", "GN"] :
        if gpu :
            model = model.cuda()
        train_model(
            model,
            train_loader=train_loader,
            epochs=1,
            lr=0.05,
            lr_decay_milestones='6,8',
            lr_decay_gamma=0.1,
            pruner=pruner,
            device= 'cuda' if gpu else 'cpu'
        )
        model = model.cpu()
    model.eval()
    ori_ops, ori_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    #ori_acc, ori_val_loss = eval(model, test_loader, device=args.device)

    progressive_pruning(pruner, model, speed_up=speed_up, example_inputs=example_inputs)
    del pruner  # remove reference
    pruned_ops, pruned_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    #pruned_acc, pruned_val_loss = eval(model, test_loader, device=args.device)
    print("Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(
                ori_size / 1e6, pruned_size / 1e6, pruned_size / ori_size * 100
            ))

def progressive_pruning(pruner, model, speed_up, example_inputs):
    model.eval()
    base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_speed_up = 1
    while current_speed_up < speed_up:
        pruner.step()
        pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
        if pruner.current_step == pruner.iterative_steps:
            break
    return current_speed_up

