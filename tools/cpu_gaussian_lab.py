"""Real CPU differentiable 3D Gaussian optimization demo, two calibrated views.
Synthetic validation scene only, not a replacement for real-photo COLMAP+CUDA.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

def render_gaussians(points, rgb, alpha_logits, views, size=24, focal=26.0, sigma=1.5):
    row, col = torch.meshgrid(
        torch.arange(size, device=points.device, dtype=points.dtype),
        torch.arange(size, device=points.device, dtype=points.dtype),
        indexing='ij')
    relative = points[None,:,:] - views[:,None,:]
    z = relative[:,:,2].clamp_min(0.2)
    px = relative[:,:,0] / z * focal + (size - 1) / 2
    py = relative[:,:,1] / z * focal + (size - 1) / 2
    dist = ((col[None,None,:,:] - px[:,:,None,None]) ** 2
            + (row[None,None,:,:] - py[:,:,None,None]) ** 2)
    weights = torch.exp(-dist / (2 * sigma ** 2)) * torch.sigmoid(alpha_logits)[None,:,None,None]
    sumweights = weights.sum(dim=1)
    color = (weights[:,:,:,:,None] * torch.sigmoid(rgb)[None,:,None,None,:]).sum(dim=1)
    return color / (1 + sumweights[:,:,:,None])

def export_ply(path, points, rgb, alpha):
    pts = points.detach().cpu().numpy()
    colors = (torch.sigmoid(rgb).detach().cpu().numpy() * 255).round().clip(0,255).astype('uint8')
    op = torch.sigmoid(alpha).detach().cpu().numpy()
    header = '\n'.join([
      'ply', 'format ascii 1.0', f'element vertex {len(pts)}',
      'property float x', 'property float y', 'property float z',
      'property float scale_0', 'property float scale_1', 'property float scale_2',
      'property float opacity', 'property uchar red', 'property uchar green',
      'property uchar blue', 'end_header']) + '\n'
    with open(path, 'w', encoding='utf8') as f:
        f.write(header)
        for p,c,a in zip(pts, colors, op):
            f.write(f'{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} -2.0 -2.0 -2.0 {a:.7f} {c[0]} {c[1]} {c[2]}\n')

def run_experiment(out_dir: Path, seed: int=42, steps: int=90):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    n = 12
    true_points = torch.empty(n, 3).uniform_(-.32, .32)
    true_points[:,2] = torch.rand(n) * .32 + 1.7
    true_rgb = torch.randn(n,3) * 1.2
    true_alpha = torch.ones(n) * .7
    cameras = torch.tensor([[-.11,0.,0.],[.11,0.,0.]], dtype=torch.float32)
    with torch.no_grad():
        target = render_gaussians(true_points,true_rgb,true_alpha,cameras).detach()
    points = (true_points + torch.randn_like(true_points) * .08).requires_grad_()
    color = (true_rgb + torch.randn_like(true_rgb) * .8).requires_grad_()
    alpha = (true_alpha + torch.randn_like(true_alpha) * .4).requires_grad_()
    optim = torch.optim.Adam([points,color,alpha],lr=.035)
    start = None
    for _ in range(steps):
        optim.zero_grad()
        predicted = render_gaussians(points,color,alpha,cameras)
        loss = torch.mean((predicted-target)**2)
        if start is None: start = loss.detach().item()
        loss.backward()
        optim.step()
    final = loss.detach().item()
    export_ply(out_dir/'learned_gaussians.ply',points,color,alpha)
    for i in range(2):
        render = render_gaussians(points,color,alpha,cameras)[i].detach().numpy()
        Image.fromarray(np.uint8(np.clip(render*255,0,255))).save(out_dir/f'render_view_{i}.png')
        Image.fromarray(np.uint8(np.clip(target[i].numpy()*255,0,255))).save(out_dir/f'target_view_{i}.png')
    outfile = out_dir/'learned_gaussians.ply'
    proof = {
        'engine':'torch autograd', 'hardware':'cpu',
        'kind':'real_cpu_differentiable_3d_gaussian_multiview_synthetic_scene',
        'seed':seed,'views':2,'gaussians':n,'steps':steps,
        'starting_mse':start,'final_mse':final,
        'improved':bool(final<start),'gaussian_file_bytes':outfile.stat().st_size,
        'gaussian_sha256':hashlib.sha256(outfile.read_bytes()).hexdigest(),
        'scope':'synthetic calibrated two-view scene; not COLMAP photographic data or gsplat CUDA'
    }
    (out_dir/'proof.json').write_text(json.dumps(proof,indent=2)+'\n', encoding='utf8')
    if not (final < start*.55 and outfile.stat().st_size > 200):
        raise RuntimeError('CPU Gaussian calibration convergence gate failed')
    return proof

if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--out',type=Path,default=Path('output/cpu_gaussian_lab'))
    cli.add_argument('--steps',type=int,default=90)
    args=cli.parse_args()
    print(json.dumps(run_experiment(args.out,steps=args.steps),indent=2))
