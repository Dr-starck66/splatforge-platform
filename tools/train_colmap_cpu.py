"""Bounded CPU Gaussian trainer for real COLMAP TXT reconstructions.
Uses sparse 3D points, camera poses and source RGB images. Experimental CPU
Gaussian raster approximation; never claim native gsplat CUDA.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path

def _data_lines(path: Path, keep_empty=False):
    with path.open(encoding='utf-8') as f:
        return [l.strip() for l in f if not l.lstrip().startswith('#') and (keep_empty or l.strip())]

def load_colmap(source: Path, side=32, max_points=96):
    import numpy as np
    from PIL import Image
    sparse = source / 'sparse' / '0'
    for name in ('cameras.txt', 'images.txt', 'points3D.txt'):
        if not (sparse / name).is_file():
            raise RuntimeError('COLMAP_TEXT_REQUIRED: missing ' + name)
    cams = {}
    for line in _data_lines(sparse / 'cameras.txt'):
        fields = line.split()
        cid, kind, w, h = int(fields[0]), fields[1], int(fields[2]), int(fields[3])
        pars = list(map(float, fields[4:]))
        if kind == 'PINHOLE' and len(pars) == 4:
            fx, fy, cx, cy = pars
        elif kind == 'SIMPLE_PINHOLE' and len(pars) == 3:
            fx, cx, cy = pars; fy = fx
        else:
            raise RuntimeError('COLMAP_CAMERA_UNSUPPORTED: ' + kind)
        cams[cid] = (w,h,fx,fy,cx,cy)
    pairs = _data_lines(sparse / 'images.txt', keep_empty=True)
    images = []
    for i in range(0,len(pairs),2):
        line = pairs[i]
        if not line: continue
        v = line.split(maxsplit=9)
        if len(v) < 10: raise RuntimeError('COLMAP_INVALID_IMAGE_POSE')
        quat = list(map(float,v[1:5]))
        trans = np.array(list(map(float,v[5:8])),dtype='float32')
        cid = int(v[8]); filename = v[9]
        if cid not in cams: raise RuntimeError('COLMAP_UNKNOWN_CAMERA')
        image_path = (source/'images'/filename).resolve()
        if not image_path.is_relative_to((source/'images').resolve()) or not image_path.is_file():
            raise RuntimeError('COLMAP_IMAGE_MISSING: '+filename)
        w,h,fx,fy,cx,cy = cams[cid]
        fx *= side/w; fy *= side/h; cx *= side/w; cy *= side/h
        qw,qx,qy,qz = quat
        rot = np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
                        [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
                        [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]],dtype='float32')
        with Image.open(image_path) as img:
            image = np.asarray(img.convert('RGB').resize((side,side),Image.Resampling.BILINEAR),dtype='float32')/255.
        images.append((rot,trans,(fx,fy,cx,cy),image,filename))
    if len(images) < 2: raise RuntimeError('COLMAP_INSUFFICIENT_REGISTERED_VIEWS')
    points=[]; colors=[]
    for line in _data_lines(sparse/'points3D.txt'):
        f=line.split()
        if len(f)<8: raise RuntimeError('COLMAP_INVALID_3D_POINT')
        points.append(list(map(float,f[1:4])))
        colors.append([int(v)/255 for v in f[4:7]])
    if len(points)<3: raise RuntimeError('COLMAP_INSUFFICIENT_SPARSE_POINTS')
    stride=max(1,math.ceil(len(points)/max_points))
    return np.array(points[::stride][:max_points],dtype='float32'),np.array(colors[::stride][:max_points],dtype='float32'),images[:12]

def train(source: Path, model_dir: Path, steps=80, side=32):
    import numpy as np
    import torch
    from PIL import Image
    if steps<10 or steps>1000 or side<16 or side>64: raise ValueError('CPU limits violated')
    torch.manual_seed(20261017)
    torch.set_num_threads(min(2,os.cpu_count() or 1))
    pts,cols,images=load_colmap(source,side=side)
    p=torch.nn.Parameter(torch.tensor(pts.copy()))
    init_color=np.clip(cols,.025,.975)
    c=torch.nn.Parameter(torch.tensor(np.log(init_color/(1-init_color)),dtype=torch.float32))
    scale=torch.nn.Parameter(torch.full((len(pts),),-2.5))
    alpha=torch.nn.Parameter(torch.full((len(pts),),1.0))
    image_data=[(torch.tensor(r),torch.tensor(t),torch.tensor(k),torch.tensor(rgb)) for r,t,k,rgb,_ in images]
    yy,xx=torch.meshgrid(torch.arange(side).float(),torch.arange(side).float(),indexing='ij')
    def render(r,t,k):
        local=p@r.T+t
        z=local[:,2].clamp_min(.05)
        px=local[:,0]/z*k[0]+k[2]
        py=local[:,1]/z*k[1]+k[3]
        sigma=(torch.exp(scale)*k[0]/z).clamp(.5,12)
        dist=((xx[None,:,:]-px[:,None,None])**2+(yy[None,:,:]-py[:,None,None])**2)
        w=torch.exp(-dist/(2*sigma[:,None,None]**2))*torch.sigmoid(alpha)[:,None,None]*(local[:,2]>0)[:,None,None]
        return (w[:,:,:,None]*torch.sigmoid(c)[:,None,None,:]).sum(0)/(1+w.sum(0)[:,:,None])
    opt=torch.optim.Adam([{'params':[p],'lr':.004},{'params':[c,scale,alpha],'lr':.045}])
    def evaluate():
        return torch.stack([((render(r,t,k)-rgb)**2).mean() for r,t,k,rgb in image_data]).mean()
    start=float(evaluate().detach().item())
    for _ in range(steps):
        opt.zero_grad()
        loss=evaluate()
        if not torch.isfinite(loss): raise RuntimeError('CPU_TRAIN_NONFINITE_LOSS')
        loss.backward()
        opt.step()
    final=float(evaluate().detach().item())
    if not (final < start*.98):
        raise RuntimeError(f'CPU_TRAIN_NO_CONVERGENCE: {start:.7f} -> {final:.7f}')
    model_dir.mkdir(parents=True,exist_ok=True)
    splat_path=model_dir/'scene.splat'
    with splat_path.open('wb') as f:
        for xyz,col,sc,a in zip(p.detach(),c.detach(),scale.detach(),alpha.detach()):
            xyz=xyz.tolist()
            side_scale=float(torch.exp(sc).item())
            rgb=[max(0,min(255,round(v*255))) for v in torch.sigmoid(col).tolist()]
            opacity=max(0,min(255,round(torch.sigmoid(a).item()*255)))
            f.write(struct.pack('<6f8B',*xyz,side_scale,side_scale,side_scale,*rgb,opacity,255,128,128,128))
    if splat_path.stat().st_size!=len(pts)*32: raise RuntimeError('BAD_SPLAT_EXPORT')
    with torch.no_grad():
        sample=render(*image_data[0][:3]).numpy()
        Image.fromarray(np.uint8(np.clip(sample*255,0,255))).save(model_dir/'render_view_0.png')
    proof={'engine':'pytorch_cpu_gaussian_colmap','status':'PASS_CPU_GAUSSIAN_SCENE',
           'provenance':'COLMAP camera poses + sparse points + input RGB images',
           'image_count':len(images),'gaussian_count':len(pts),'steps':steps,
           'initial_mse':start,'final_mse':final,'artifact_bytes':splat_path.stat().st_size,
           'artifact_sha256':hashlib.sha256(splat_path.read_bytes()).hexdigest(),
           'limits':'bounded CPU experimental forward renderer, NOT gsplat CUDA or full-quality 3DGS'}
    (model_dir/'proof.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    return proof

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-s','--source',type=Path,required=True)
    parser.add_argument('-m','--model',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=80)
    args=parser.parse_args()
    print(json.dumps(train(args.source,args.model,steps=args.steps),indent=2))
