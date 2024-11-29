from utils.util import get_run_name
import argparse, copy
import numpy as np
from dl.cifar100 import cifar100
from eval.torch_eval import TorchEvaluator
from prune.prunes import *
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action="store_true", default=True)
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--orig-model', type=str, default="./pmodels/resnet50_fx_model.pt")
parser.add_argument('--data-dir', type=str, default="./data")
parser.add_argument('--outp-dir', type=str, default="./pmodels")

parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--val-interval', type=int, default=1)

parser.add_argument('--prune-amount', type=float, default=0.3)
parser.add_argument('--prune-method', type=str, default="GN")
parser.add_argument('--speed-up', type=float, default=2.11)

parser.add_argument('--kd-method', type=int, default=1)

parser.add_argument('--wandb_log', type=int, default=0)
parser.add_argument('--wandb_project', type=str, default='')
parser.add_argument('--wandb_entity', type=str, default='')
parser.add_argument('--wandb_dir', type=str, default='')


args = parser.parse_args()
run_name = get_run_name(args);

###########################################
# Fix the random seeds
###########################################
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

teval = TorchEvaluator(args.cuda);

from kd.kds import KD_pred
if args.kd_method == 0:
    kd_all = [KD_pred]
elif args.kd_method == 1 :
    from kd.kds import KD_fit_BN
    kd_all = [KD_fit_BN, KD_pred]
elif args.kd_method == 2:
    from kd.kds import KD_fit_RL
    kd_all = [KD_fit_RL, KD_pred]
elif args.kd_method == 3:
    from kd.kds import KD_fit_GRAM
    kd_all = [KD_fit_GRAM, KD_pred]


c100 = cifar100(data_dir=args.data_dir);

train_loader = c100.load_cifar100(is_train=True,
                             batch_size=args.batch_size, num_workers= args.num_workers);
test_loader = c100.load_cifar100(is_train=False,
                             batch_size=args.batch_size, num_workers= args.num_workers);


teacher_model = torch.load(args.orig_model);
teacher_model.eval()
stud_model = copy.deepcopy(teacher_model)

global_prunes(stud_model, pmethod=args.prune_method, kmethod=args.kd_method,
                           pamount=args.prune_amount, speed_up=args.speed_up,
                          train_loader = train_loader, gpu=args.cuda)

if args.cuda :
    stud_model = stud_model.cuda()
    teacher_model = teacher_model.cuda()

teacher_model.eval()
stud_model.train()

optimizer = torch.optim.SGD(stud_model.parameters(),
                            lr=0.05,
                            momentum=0.9,
                            weight_decay=1e-4,
                            nesterov=True)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch)


if args.wandb_log:
    '''
    Initialize WanDB
    '''
    import wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name=run_name, dir=args.wandb_dir)

for ki, kdm in enumerate(kd_all) :
    kdm_i = kdm(stud_model, teacher_model);
    with tqdm(desc="Teaching ...", total=args.epochs*len(kd_all)) as pbar:
        for ep in range(args.epochs):
            loss = kdm_i.iterate_teaching(train_loader, optimizer, scheduler, gpu=args.cuda);

            if ep % args.val_interval == 0 :
                stud_model.eval()
                test_acc = teval.eval_acc(test_loader, stud_model)
                if args.wandb_log :
                    wandb.log({"loss" : loss, "test_acc" : test_acc}, step=ep + args.epochs*ki)
                else :
                    print("loss : {:.3f}, {:.2f} %".format(loss, test_acc))
                stud_model.train()

            pbar.update(1);
    kdm_i.remove_all_hooks()
    torch.save(stud_model, os.path.join(args.outp_dir, run_name + " Step {}.pt".format(ki+1)))

