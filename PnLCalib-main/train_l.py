import sys
import yaml
import wandb
import torch
import argparse
import warnings
import numpy as np
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from utils.utils_train_l import train_one_epoch, validation_step
from model.dataloader_l import SoccerNetCalibrationDataset, WorldCup2014Dataset, TSWorldCupDataset
from model.cls_hrnet_l import get_cls_net
from model.losses import MSELoss

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.RankWarning)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True,
                        help="Path to the configuration file")
    parser.add_argument("--dataset", type=str, default='SoccerNet',
                        help="Dataset name (SoccerNet, WorldCup14, TSWorldCup) (default: SoccerNet)")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Save directory")
    parser.add_argument("--cuda", type=str, default="cuda:0",
                        help="CUDA device index (default: 'cuda:0')")
    parser.add_argument("--batch", type=int, default=2,
                        help="Batch size for train / val (default: 2)")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for data loading (default: 4)")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--pretrained", type=str, default='',
                        help="Pretrained weights path (default: '')")
    parser.add_argument("--lr0", type=float, default=0.001,
                        help="Initial learning rate (default: 0.001)")
    parser.add_argument("--patience", type=int, default=8,
                        help="Patience parameter for lr scheduler (default: 8)")
    parser.add_argument("--factor", type=float, default=0.5,
                        help="Reducing factor for lr scheduler (default: 0.5)")
    parser.add_argument("--transform", type=bool, default=True,
                        help="Apply data augmentation (default: True)")
    parser.add_argument("--wandb_project", type=str, default='',
                        help="Wandb project name")

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    wandb.login()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    run = wandb.init(
        mode="online" if args.wandb_project != '' else "offline",
        project=args.wandb_project,
        config={
            "batch": args.batch,
            "learning_rate_0": args.lr0,
            "patience": args.patience,
            "factor": args.factor,
            "epochs": args.num_epochs,
            "pretrained": args.pretrained,
            "time": timestamp
        })

    dataset = args.dataset
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    cfg = yaml.safe_load(open(args.cfg, 'r'))

    if dataset == "SoccerNet":
        from model.transforms_l import transforms_w_LR, no_transforms_w_LR
        training_set = SoccerNetCalibrationDataset(args.root_dir, "train", transform=transforms_w_LR,
                                                   main_cam_only=False)
        validation_set = SoccerNetCalibrationDataset(args.root_dir, "valid", transform= no_transforms_w_LR,
                                                     main_cam_only=False)
    elif dataset == "WorldCup14":
        from model.transformsWC_l import transforms, no_transforms
        training_set = WorldCup2014Dataset(args.root_dir, "train_val",
                                           transform=transforms if args.transform else no_transforms)
        validation_set = WorldCup2014Dataset(args.root_dir, "test", transform= no_transforms)

    elif dataset == "TSWorldCup":
        from model.transformsWC_l import transforms, no_transforms
        training_set = TSWorldCupDataset(args.root_dir, "train",
                                           transform=transforms if args.transform else no_transforms)
        validation_set = TSWorldCupDataset(args.root_dir, "test", transform= no_transforms)
    else:
        sys.exit("Wrong dataset name. Options: [SoccerNet, WorldCup2014, TSWorldCup]")

    training_loader = torch.utils.data.DataLoader(training_set, num_workers=args.num_workers, batch_size=args.batch,
                                                  shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, num_workers=args.num_workers, batch_size=args.batch,
                                                    shuffle=False)

    model = get_cls_net(cfg)
    if args.pretrained != "":
        loaded_state = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(loaded_state)
    model.to(device)


    loss_fn = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, mode='min', \
                                                        factor=args.factor)

    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    best_vloss = np.inf
    loss_counter = 0

    for epoch in range(args.num_epochs):
        avg_loss = train_one_epoch(epoch+1, training_loader, optimizer, loss_fn, model, device, dataset)
        avg_vloss, acc, prec, rec, f1 = validation_step(validation_loader, loss_fn, model, device, dataset)
        scheduler.step(avg_vloss)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print(f'Acc: {round(acc,3)}  Prec: {round(prec,3)}  Rec: {round(rec,3)}  F1: {round(f1,3)}')

        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss},
                        epoch_number + 1)
        writer.flush()

        wandb.log({"train_loss": avg_loss, "val_loss": avg_vloss, "epoch": epoch+1,
                    'lr': optimizer.param_groups[0]["lr"], 'Accuracy': acc, 'Precision': prec, 'Recall': rec,
                    'F1-score': f1})

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = args.save_dir + '_{}'.format(timestamp)
            torch.save(model.state_dict(), model_path)
            loss_counter = 0
        else:
            loss_counter += 1

        if loss_counter == 16:
            print('Early stopping at epoch {}'.format(epoch_number + 1))
            break

        epoch_number += 1
