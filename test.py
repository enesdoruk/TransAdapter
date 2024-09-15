import argparse
import torch
import numpy as np
from tqdm import tqdm

from utils.transforms import get_transform
from utils.data_list_image import ImageList
from models.modelling_transadapter import TransAdapter
from utils.utils import set_seed, simple_accuracy, visda_acc, load_pretrained

import warnings
warnings.filterwarnings("ignore")



def test(args, model):
    _, _ , transform_test = get_transform(args.dataset, args.img_size)

    test_loader = torch.utils.data.DataLoader(
        ImageList(open(args.test_list_target).readlines(), transform=transform_test, mode='RGB'),
        batch_size=args.eval_batch_size, shuffle=False, num_workers=4)

    model.eval()

    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            out = model(x)

            preds = torch.argmax(out[0], dim=-1)

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

    all_preds, all_label = all_preds[0], all_label[0]
    if args.dataset == 'visda17':
        accuracy, classWise_acc = visda_acc(all_preds, all_label)
    else:
        accuracy = simple_accuracy(all_preds, all_label)

    if args.dataset == 'visda17':
        return accuracy, classWise_acc
    else:
        return accuracy, None


def get_args():
    parser = argparse.ArgumentParser(description='Train the Unsupervised domain adaptation')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--test_list_target', type=str, default=None, help='test list target data')
    parser.add_argument('--img_size', type=int, default=256, help='image size')
    parser.add_argument('--window_size', type=int, default=8, help='Window size')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='evaluation batch')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--num_classes", default=10, type=int, help="Number of classes in the dataset.")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--name", required=True, help="Name of this run. Used for monitoring.")
    parser.add_argument('--pretrained_dir', type=str, default=None, help='pretrained weight dir')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    model = TransAdapter(img_size=args.img_size, window_size=args.window_size, num_classes=args.num_classes)

    if args.pretrained_dir is not None:
        model = load_pretrained(args.pretrained_dir, model)

    model.to(args.device)

    if args.dataset == 'visda17':
        accuracy, classWise_acc = test(args, model)
    else:
        accuracy, _ = test(args, model)

    print("Accuracy: ", accuracy)