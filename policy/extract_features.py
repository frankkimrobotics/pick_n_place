"""extract_features :: frozen-encoder patch tokens for every dataset frame.

Encoders (all ViT-B/16 class, frozen):
  clip    openai/clip-vit-base-patch16
  dinov2  facebook/dinov2-base (ViT-B/14, 256 patch tokens)
  lingbot robbyant/lingbot-vision-vit-base

Input: wrist D405 RGB mp4 (432x240) -> resize height to 224, center crop
224x224 (UMI ImgRes). Saves per scene <enc>_tokens.npz: fp16 [T, P, D] patch
tokens (no CLS); pooled features are derived as token means at train time.

Run in the ilpolicy env:  python extract_features.py --enc clip
"""
import argparse
import glob
import os

import numpy as np
import torch
import imageio.v3 as iio

DS = os.path.expanduser("~/pnp_dataset")
DEV = "cuda:0"


def load_encoder(name):
    from transformers import AutoModel, AutoImageProcessor, CLIPVisionModel
    if name == "clip":
        mdl = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        fwd = lambda px: mdl(pixel_values=px).last_hidden_state[:, 1:]
    elif name == "dinov2":
        mdl = AutoModel.from_pretrained("facebook/dinov2-base")
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        def fwd(px):
            # drop CLS: dinov2-base at 224px -> 16x16 = 256 patch tokens
            return mdl(pixel_values=px).last_hidden_state[:, 1:]
    elif name == "lingbot":
        from lingbot_vision import load_pretrained_backbone
        mdl, dim = load_pretrained_backbone(variant="base", device=DEV,
                                            dtype=torch.float32, verbose=False)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        def fwd(px):
            return mdl(px, is_training=True)["x_norm_patchtokens"]
    else:
        raise ValueError(name)
    mdl.eval().to(DEV)
    for p in mdl.parameters():
        p.requires_grad_(False)
    return fwd, torch.tensor(mean).view(1, 3, 1, 1), torch.tensor(std).view(1, 3, 1, 1)


def frames_224(mp4):
    fr = iio.imread(mp4)                      # [T, 240, 432, 3]
    t = torch.from_numpy(fr).permute(0, 3, 1, 2).float() / 255.0
    t = torch.nn.functional.interpolate(t, size=(224, 404), mode="bilinear",
                                        align_corners=False)
    x0 = (404 - 224) // 2
    return t[:, :, :, x0:x0 + 224]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True, choices=["clip", "dinov2", "lingbot"])
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    fwd, mean, std = load_encoder(a.enc)
    mean, std = mean.to(DEV), std.to(DEV)
    scenes = sorted(glob.glob(os.path.join(DS, "scene_*")))
    for sdir in scenes:
        out = os.path.join(sdir, f"{a.enc}_tokens.npz")
        if os.path.exists(out):
            continue
        imgs = frames_224(os.path.join(sdir, "d405_rgb.mp4"))
        toks = []
        with torch.no_grad():
            for i in range(0, len(imgs), a.batch):
                px = ((imgs[i:i + a.batch].to(DEV)) - mean) / std
                toks.append(fwd(px).half().cpu())
        toks = torch.cat(toks).numpy()
        np.savez(out, tokens=toks)
        print(os.path.basename(sdir), toks.shape)
    print("done", a.enc)


if __name__ == "__main__":
    main()
