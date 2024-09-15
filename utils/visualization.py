import cv2
import torch
import wandb
import numpy as np  
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image



def tsne_visualization(source_feats, target_feats, mode=''):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    
    combined_feats = torch.cat((source_feats, target_feats), dim=0)
    tsne = tsne.fit_transform(combined_feats.detach().cpu().numpy())

    x_min, x_max = np.min(tsne, 0), np.max(tsne, 0)
    tsne = (tsne - x_min) / (x_max - x_min)

    source_domain_list = torch.zeros(source_feats.shape[0]).type(torch.LongTensor)
    target_domain_list = torch.ones(target_feats.shape[0]).type(torch.LongTensor)
    combined_domain_list = torch.cat((source_domain_list, target_domain_list), 0).cuda()

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for i in range(len(combined_domain_list)): 
        if combined_domain_list[i] == 0:
            colors = 'blue'
        else:
            colors = 'red'
        ax.scatter(tsne[i,0], tsne[i,1], color=colors, )

    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

    images = wandb.Image(image, caption=mode)
    wandb.log({"TSNE Visualization": images})



def attention_visualization(img, attns, name=''):
    wandb_images = []

    for attn in attns:
        for i, (im, att) in enumerate(zip(img, attn)):
            img_np = im.cpu().detach().numpy().transpose(1,2,0)*255
            img_np = img_np.astype('uint8')
            
            mean_att = torch.mean(att, dim=2)
            mean_att = mean_att.cpu().detach().numpy()

            mean_att *= 255
            mean_att = 255 - mean_att.astype('uint8')

            heatmap = cv2.applyColorMap(mean_att, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (img_np.shape[0], img_np.shape[1]))

            heatmap = cv2.addWeighted(heatmap, 0.7, img_np, 0.3, 0)
            
            wandb_image = wandb.Image(heatmap, caption=f"attention_{name}", file_type="jpg")
            wandb_images.append(wandb_image)
        break

    wandb.log({'Attention Visualization': wandb_images})


def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def get_gradCAM(x, model, name=''):
    target_layer = [model.transformer.layers[-1].blocks[-1].norm1]
    cam = GradCAM(model=model, target_layers=target_layer,use_cuda=True, reshape_transform=reshape_transform)
    cam.batch_size = x.size(0)
    grayscale_cam = cam(input_tensor=x, targets=None)

    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(x[0].cpu().detach().numpy().transpose(1,2,0), grayscale_cam, use_rgb=True)
    wandb_image = wandb.Image(cam_image, caption=f"gradCAM_{name}", file_type="jpg")

    wandb.log({'GRADCAM Visualization': wandb_image})