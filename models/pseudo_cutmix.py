import torch 
from torchvision.transforms import v2
from utils.config import get_swin_base
from models.modelling_baseline import SwinTransformer
from utils.utils import load_pretrained_psedo

  

class PseudoFeatMix:
    def __init__(self, args, logger, thresh=0.5, pseudo=False):
        super(PseudoFeatMix).__init__()

        self.args = args
        self.logger = logger
        self.thresh = thresh
        self.pseudo = pseudo

        model_config = get_swin_base()
        self.model = SwinTransformer(model_config)
        self.model.to(args.device)

        if args.psedo_ckpt is not None and self.pseudo:
            load_pretrained_psedo(self.args.psedo_ckpt, self.model, logger)

        cutmix = v2.CutMix(alpha=1., num_classes=self.args.num_classes)
        mixup = v2.MixUp(alpha=1., num_classes=self.args.num_classes)
        self.feat_cmix = v2.RandomChoice([cutmix, mixup])

    
    def classify(self, target_loader):
        self.model.eval()
        labels = {}

        for batch in target_loader:
            x, y, ind = batch[0], batch[1], batch[2]
            x, y = x.to(self.args.device), y.to(self.args.device)
           
            with torch.no_grad():
                logits, _, _ = self.model(x)
                act = torch.nn.functional.softmax(logits)
                pred = torch.argmax(act, dim=-1)

                for i in range(pred.shape[0]):
                    if torch.max(act[i]) > self.thresh:
                        labels[ind[i].item()] = pred[i].cpu().detach().item()
                    else:
                        labels[ind[i].item()] = -1

        return labels

    def run(self, source, target):
        if self.pseudo:
            tp_ind = torch.where(target[2] != -1)[0]
            fp_ind = torch.where(target[2] == -1)[0]

            x_gt, y_gt = source[0][tp_ind], source[1][tp_ind]
            xt_gt, yt_gt = target[0][tp_ind], target[1][tp_ind]

            x, y = self.feat_cmix(torch.cat((x_gt, xt_gt), dim=0), \
                    torch.cat((y_gt, yt_gt),dim=0))

            x_sc = x[:x_gt.size(0),:,:,:]
            y_sc = torch.argmax(y[:y_gt.size(0),:], dim=1)

            x_sc, y_sc = torch.cat((x_sc, source[0][fp_ind]), dim=0), torch.cat((y_sc, source[1][fp_ind]), dim=0)
        else:
            x, y = self.feat_cmix(torch.cat((source[0], target[0]), dim=0), \
                       torch.cat((source[1], target[1]),dim=0))

            x_sc = x[:source[0].size(0),:,:,:]
            y_sc = torch.argmax(y[:source[0].size(0),:], dim=1)


        return x_sc, y_sc