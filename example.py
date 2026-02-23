import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


from carl.trainer.seg_trainer import LinearTrainer


def load_model(cfg):
    with open(cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    ckpt_path = cfg["training_kwargs"]["ssl_ckpt_path"]
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    print(cfg)

    model = LinearTrainer(cfg)
    missing, unexecpted = model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.to("cuda", dtype=torch.float32)
    model.eval()

    return model

def load_data(img_path, wlens_path):
    img = torch.from_numpy(np.load(img_path)).unsqueeze(0).float().cuda()
    img = (img - img.mean()) / img.std()

    wlens = np.loadtxt(wlens_path)
    wlens = torch.from_numpy(wlens).float().unsqueeze(0)
    wlens = wlens / 1000.0  # convert to micrometers
    wlens = wlens.cuda()

    return img, wlens

def plot_features(img, spat_representations, spec_representations, out_file, rgb_channels = None):

    d,h,w = spat_representations.shape
    spat_representations = spat_representations.permute(1,2,0).flatten(0,1)
    u,s,v = torch.pca_lowrank(spat_representations, q=3)
    s = torch.diag(s)
    pca = torch.matmul(u, s)
    pca = pca.reshape(h, w, 3).permute(2,0,1).unsqueeze(0)
    pca = F.interpolate(pca, size=img.shape[1:], mode='bilinear', align_corners=False)
    pca = pca[0].permute(1,2,0).cpu().numpy()

    d,h,w = spec_representations.shape
    spec_representations = spec_representations.permute(1,2,0).flatten(0,1)
    u,s,v = torch.pca_lowrank(spec_representations, q=3)
    s = torch.diag(s)
    pca_spec = torch.matmul(u, s)
    pca_spec = pca_spec.reshape(h, w, 3).permute(2,0,1).unsqueeze(0)
    pca_spec = F.interpolate(pca_spec, size=img.shape[1:], mode='bilinear', align_corners=False)
    pca_spec = pca_spec[0].permute(1,2,0).cpu().numpy()

    fig, ax = plt.subplots(1,3, figsize=(15,6))

    img = img.permute(1,2,0).cpu().numpy()
    if rgb_channels is None:
        rgb_channels = np.linspace(0, img.shape[-1]-1, 3)
        rgb_channels = rgb_channels.astype(int).tolist()
    img = img[..., rgb_channels]
    img = (img - img.min()) / (img.max() - img.min())
    ax[0].imshow(img)
    ax[0].set_title("Input Image")

    feats = (pca - pca.min()) / (pca.max() - pca.min())
    ax[1].imshow(feats)
    ax[1].set_title("Spatial Features")

    feats_spec = (pca_spec - pca_spec.min()) / (pca_spec.max() - pca_spec.min())
    ax[2].imshow(feats_spec)
    ax[2].set_title("Spectral Features")

    for a in ax:
        a.axis('off')

    plt.savefig(out_file, bbox_inches='tight')
    plt.close()

    return


if __name__ == "__main__":

    base_path = os.path.dirname(os.path.abspath(__file__))

    cfg = base_path + "/configs/config_seg.yaml"

    model = load_model(cfg)

    img_path = base_path  + "/example_data/img.npy" # add some example enmap data here
    wlens_path = base_path  + "/example_data/enmap_202.txt"
    img, wlens = load_data(img_path, wlens_path)

    with torch.no_grad():
        spat_representations, spec_representations = model.model(img, wlens)

    out_path = base_path + "/example_data/feature_plot.png"
    rgb_channels = [54, 28, 15]
    plot_features(img[0], spat_representations[0], spec_representations[0], out_path, rgb_channels)

