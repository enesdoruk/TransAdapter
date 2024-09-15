import torch
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DataParallel

import os
import time
import datetime
import argparse
import numpy as np
import torch.utils
import torch.utils.data
from tqdm import tqdm

from utils.logger import create_logger
from utils.config import get_transadapter_base, get_sweep
from utils.transforms import get_transform
from models.modelling_transadapter import TransAdapter
from models.pseudo_cutmix import PseudoFeatMix
from models.graph_alignment import GraphConvDiscriminator
from utils.data_list_image import ImageList, ImageListIndex
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.utils import (set_seed, AverageMeter, save_model, simple_accuracy, \
                        visda_acc, count_parameters, load_pretrained, log_image_table, \
                        get_grad_norm)
from utils.visualization import tsne_visualization, attention_visualization, get_gradCAM
from utils.lossZoo import adv_global, FocalLoss

import warnings
warnings.filterwarnings("ignore")

import wandb 


def valid(args, model, test_loader, iter, ad_net_local, mode=''):
    eval_losses = AverageMeter()

    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=False,
                          disable=args.local_rank not in [-1, 0])

    loss_fct = CrossEntropyLoss()

    tensors = []
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits, feats, _, _, _, attn_mats, _,  _, _ = model(x, ad_net=ad_net_local, is_train=False)

            tensors.append(feats)

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            wandb.log({"val/loss": eval_loss.item()})

            preds = torch.argmax(logits, dim=-1)

            if step % 50 == 0 and mode == 'target':
                log_image_table(x, preds, y)
                attention_visualization(x, attn_mats, name=f'{step}_{iter}')

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    cat_tens = torch.cat(tensors, dim=0)

    all_preds, all_label = all_preds[0], all_label[0]
    if args.dataset == 'visda17':
        accuracy, classWise_acc = visda_acc(all_preds, all_label)
    else:
        accuracy = simple_accuracy(all_preds, all_label)

    if args.dataset == 'visda17':
        return accuracy, classWise_acc, cat_tens
    else:
        return accuracy, None, cat_tens
    

def train(args, model, ad_net_local, ad_net_global, logger):    
    transform_source, transform_target, transform_test = get_transform(args.dataset, args.img_size)

    source_loader = torch.utils.data.DataLoader(
        ImageList(open(args.source_list).readlines(), transform=transform_source, mode='RGB'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    target_loader = torch.utils.data.DataLoader(
        ImageListIndex(open(args.target_list).readlines(), transform=transform_target, mode='RGB'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=4, drop_last=True)

    test_loader_source = torch.utils.data.DataLoader(
        ImageList(open(args.test_list_source).readlines(), transform=transform_test, mode='RGB'),
        batch_size=args.eval_batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    test_loader_target = torch.utils.data.DataLoader(
        ImageList(open(args.test_list_target).readlines(), transform=transform_test, mode='RGB'),
        batch_size=args.eval_batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    
    if args.n_gpu > 1:
        optimizer = torch.optim.SGD([
                                        {'params': model.module.transformer.parameters()},
                                        {'params': model.module.head.parameters()},
                                    ],
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        
        optimizer_ad = torch.optim.SGD(list(ad_net_local.module.parameters()) + list(ad_net_global.module.parameters()),
                            lr=args.learning_rate/10, 
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD([
                                        {'params': model.transformer.parameters()},
                                        {'params': model.head.parameters()},
                                    ],
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        
        optimizer_ad = torch.optim.SGD(list(ad_net_local.parameters()) + list(ad_net_global.parameters()),
                            lr=args.learning_rate/10, 
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    

    t_total = args.num_steps

    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler_ad = WarmupCosineSchedule(optimizer_ad, warmup_steps=args.warmup_steps, t_total=t_total)

    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler_ad = WarmupLinearSchedule(optimizer_ad, warmup_steps=args.warmup_steps, t_total=t_total)

    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    ad_net_local.zero_grad()
    ad_net_global.zero_grad()

    set_seed(args)
    
    losses = AverageMeter()
    best_classWise_acc_source, best_acc_source = '', 0
    best_classWise_acc_target, best_acc_target = '', 0

    len_source = len(source_loader)
    len_target = len(target_loader)            

    training_epoch_iterator = tqdm(range(1, t_total),
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=False,
                          disable=args.local_rank not in [-1, 0])
    
    logger.info("Start training")
    start_time = time.time()

    pseudomix = PseudoFeatMix(args, logger, 0.5, args.pseudo_lab)
    if args.pseudo_lab:
        pseudo_labels = pseudomix.classify(target_loader)

    for i, global_step in enumerate(training_epoch_iterator):
        model.train()
        ad_net_local.train()
        ad_net_global.train()

        if (global_step-1) % (len_source-1) == 0:
            iter_source = iter(source_loader)    
        if (global_step-1) % (len_target-1) == 0:
            iter_target = iter(target_loader)

        data_source = next(iter_source)
        data_target = next(iter_target)
   
        x_s, y_s = tuple(t.to(args.device) for t in data_source)
        target = tuple(t.to(args.device) for t in data_target)

        if args.pseudo_lab:
            pseudo = []
            for i in range(target[0].shape[0]):
                try:
                    pseudo.append(pseudo_labels[target[2][i].item()])
                except:
                    pseudo.append(target[1][i].item())
            target[2][:] = torch.from_numpy(np.array(pseudo)).to(args.device)

        x_t = target[0]
        x_s, y_s = pseudomix.run((x_s,y_s), target)

        loss_fct = CrossEntropyLoss()

        logits_s, feats, feats_glb, feats_t, feats_t_glb, attn_mats, attn_mats_t, \
            loss_ad_local, kl_div = model(x_s, x_t, ad_net_local, is_train=True)
        
        foc_loss = FocalLoss(args.num_classes)
        loss_ad_global  = adv_global(foc_loss, feats_glb, feats_t_glb, ad_net_global)
        
        if i+1 % 50 == 0:
            get_gradCAM(x_t, model,  name=f'{i}_{iter}')

        loss_clc = loss_fct(logits_s.view(-1, args.num_classes), y_s.view(-1))

        loss = loss_clc + args.gamma * loss_ad_local + loss_ad_global * args.beta + kl_div * args.theta
        losses.update(loss.item())

        loss.backward()

        wandb.log({"train/loss": loss_clc.item()})
        wandb.log({"train/loss_local_ad": loss_ad_local.item()*args.gamma})

        training_epoch_iterator.set_description("Training (%d / %d steps)... (cls=%2.5f)  (ad_local=%2.5f)  (ad_global=%2.5f) (kl_div=%2.5f)" \
                                                 % (global_step, t_total, loss_clc.item(), (args.gamma * loss_ad_local).item(), \
                                                    (args.beta * loss_ad_global).item(), (args.theta * kl_div).item()))

        total_norm = get_grad_norm(model.parameters())
        total_norm_ad_local = get_grad_norm(ad_net_local.parameters())
        
        wandb.log({"train/grad_norm": total_norm})
        wandb.log({"train/grad_norm_ad_local": total_norm_ad_local})

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(ad_net_local.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(ad_net_global.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        optimizer_ad.step()
        optimizer_ad.zero_grad()
        scheduler_ad.step()

        wandb.log({"train/lr": scheduler.get_last_lr()[0]})
        wandb.log({"train/lr_local_ad": scheduler_ad.get_last_lr()[0]})
        
        if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
            model.eval()
            ad_net_local.eval()

            accuracy_source, classWise_acc_source, source_test_feats = valid(args, model, test_loader_source, global_step, ad_net_local, mode='source')
            if best_acc_source < accuracy_source:
                best_acc_source = accuracy_source

                if classWise_acc_source is not None:
                    best_classWise_acc_source = classWise_acc_source

            accuracy_target, classWise_acc_target, target_test_feats = valid(args, model, test_loader_target, global_step, ad_net_local, mode='target')
            if best_acc_target < accuracy_target:
                best_acc_target = accuracy_target
                save_model(args, model, 'transadaper_wa')

                if classWise_acc_target is not None:
                    best_classWise_acc_target = classWise_acc_target

            tsne_visualization(source_test_feats, target_test_feats, mode=f'validation_{global_step}')

            model.train()
            ad_net_local.train()
            ad_net_global.train()

            logger.info("Current Best Accuracy for source: %2.5f" % best_acc_source)
            logger.info("Current Best Accuracy for target: %2.5f" % best_acc_target)
            
            wandb.log({"val/source_accuracy": best_acc_source})
            wandb.log({"val/target_accuracy": best_acc_target})

            if args.dataset == 'visda17':
                logger.info("Current Best element-wise acc for source: %s" % best_classWise_acc_source)
                logger.info("Current Best element-wise acc for target: %s" % best_classWise_acc_target)


    logger.info("Best Accuracy for source: \t%f" % best_acc_source)
    logger.info("Best Accuracy for target: \t%f" % best_acc_target)
    if args.dataset == 'visda17':
        logger.info("Best element-wise Accuracy for source: \t%s" % best_classWise_acc_source)
        logger.info("Best element-wise Accuracy for target: \t%s" % best_classWise_acc_target)
    logger.info("End Training!")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    

def get_args():
    parser = argparse.ArgumentParser(description='Train the Unsupervised domain adaptation')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--source_list', type=str, default=None, help='source list')
    parser.add_argument('--target_list', type=str, default=None, help='target list')
    parser.add_argument('--test_list_source', type=str, default=None, help='test list source data')
    parser.add_argument('--test_list_target', type=str, default=None, help='test list target data')
    parser.add_argument('--num_steps', type=int, default=10000, help='number of steps')
    parser.add_argument('--train_batch_size', type=int, default=8, help='train batch')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='evaluation batch')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--learning_rate", default=3e-2, type=float, help="The initial learning rate for SGD.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine", help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int, help="Step of training to perform learning rate warmup for.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--name", required=True, help="Name of this run. Used for monitoring.")
    parser.add_argument('--pretrained_dir', type=str, default=None, help='pretrained weight dir')
    parser.add_argument('--psedo_ckpt', type=str, default=None, help='pretrained weight dir')
    
    parser.add_argument("--sweep", type=bool, default=False, help="enable or disable sweep")
    parser.add_argument("--num_classes", default=31, type=int, help="number of classes")
    parser.add_argument("--model", default='base', type=str, help="model type (tiny, small, base, large)")
    parser.add_argument("--wandb_name", default='', type=str, help="wandb name")
    parser.add_argument("--gamma", default=0.1, type=float,  help="The importance of the local adversarial loss.")
    parser.add_argument("--beta", default=0.1, type=float,  help="The importance of the global adversarial loss.")
    parser.add_argument("--theta", default=0.1, type=float,  help="The importance of the global mmd")
    parser.add_argument('--pseudo_lab', type=bool, default=False, help='Psuedo label usage')

        
    return parser.parse_args()


def main(logger):
    args = get_args()

    wandb.init(project="Unsupervised Vision Transformer", name=f'{args.wandb_name}')

    logger.info(f"Creating model:{args.model}")
    
    model_config = get_transadapter_base()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    
    args.device = device
    args.img_size = model_config.IMG_SIZE
    model_config.NUM_CLASSES = args.num_classes
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if args.sweep:
        args.train_batch_size = wandb.config.train_batch_size
        args.learning_rate = wandb.config.learning_rate
        args.num_steps = wandb.config.num_steps
        args.eval_batch_size = wandb.config.eval_batch_size
        args.weight_decay = wandb.config.weight_decay
        args.decay_type = wandb.config.decay_type
        args.warmup_steps = wandb.config.warmup_steps
        args.max_grad_norm = wandb.config.max_grad_norm
        args.momentum = wandb.config.momentum
        args.num_classes = wandb.config.num_classes
        args.beta = wandb.config.beta
        args.gamma = wandb.config.gamma
        args.pretrained_dir = wandb.config.pretrained_dir
        args.num_steps = wandb.config.num_steps
        args.dataset = wandb.config.dataset
        args.source_list = wandb.config.source_list
        args.target_list = wandb.config.target_list
        args.test_list_source = wandb.config.test_list_source
        args.test_list_target = wandb.config.test_list_target

        model_config.IMG_SIZE = wandb.config.img_size
        model_config.PATCH_SIZE = wandb.config.patch_size
        model_config.IN_CHANS = wandb.config.in_chans
        model_config.NUM_CLASSES = wandb.config.num_classes
        model_config.EMBED_DIM = wandb.config.embed_dim
        model_config.DEPTHS = wandb.config.depths
        model_config.NUM_HEADS = wandb.config.num_heads
        model_config.WINDOW_SIZE = wandb.config.window_size
        model_config.MLP_RATIO = wandb.config.mlp_ratio
        model_config.QKV_BIAS = wandb.config.qkv_bias
        model_config.QK_SCALE = wandb.config.qk_scale
        model_config.DROP_RATE = wandb.config.drop_rate
        model_config.ATTN_DROP_RATE = wandb.config.attn_drop_rate
        model_config.DROP_PATH_RATE = wandb.config.drop_path_rate
        model_config.APE = wandb.config.ape
        model_config.PATCH_NORM = wandb.config.patch_norm
        model_config.USE_CHECKPOINT = wandb.config.use_checkpoint
        model_config.FUSED_WINDOW_PROCESS = wandb.config.fused_window_process
        model_config.SHIFT_SIZE = wandb.config.shift_size

    set_seed(args)

    model = TransAdapter(model_config)
    model.to(args.device)
    logger.info(str(model))

    ad_net_local = GraphConvDiscriminator(in_features=32,in_dim=64*49*32, out_dim=49*49, \
                                          drop_rat=0.1, n=args.train_batch_size, pool_shape=(49,32))
    ad_net_local.to(args.device)

    ad_net_global = GraphConvDiscriminator(in_features=1024, in_dim=7*7*1024, out_dim=7*7, \
                                           drop_rat=0.1, n=args.train_batch_size, pool_shape=(7,1024))
    ad_net_global.to(args.device)

    if args.n_gpu > 1:
        model = DataParallel(model, device_ids=[i for i in range(args.n_gpu)])
        logger.info("Dataparallel is used")

    if hasattr(model.transformer, 'flops'):
        flops = model.transformer.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    params = count_parameters(model)
    logger.info(f"number of params for transformer: {params}")

    adv_params = count_parameters(ad_net_local)
    logger.info(f"number of params for adv_net: {adv_params}")


    logger.info(f'''Starting training:
        Number of GPU: {args.n_gpu}
        Dataset: {args.dataset}
        Source list: {args.source_list}   
        Target list: {args.target_list}
        Number of classes: {args.num_classes}
        Number of steps: {args.num_steps}
        Batch size: {args.train_batch_size}
        Learning rate: {args.learning_rate}
        Device: {args.device}
        Model params: {params}
        Adverserial model params: {adv_params}
    ''')

    wandb.watch(model)

    if args.pretrained_dir is not None:
        load_pretrained(args.pretrained_dir, model, logger)
    
    train(args, model, ad_net_local, ad_net_global, logger)
    

if __name__ == "__main__":
    arg = get_args()
    
    os.makedirs(f"{arg.output_dir}/{arg.dataset}", exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    logger = create_logger(output_dir='logs', name=f"{arg.model}_{str(datetime.datetime.today().strftime('_%d-%m-%H'))}")

    if arg.sweep:
        sweep_configuration = get_sweep()
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="Unsupervised Vision Transformer")

        wandb.agent(sweep_id, function=main, count=4)
    else:
        main(logger)