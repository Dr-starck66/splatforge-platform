#!/usr/bin/env python3
"""Kaggle GPU capability probe for SplatForge; not 3D reconstruction proof."""
import argparse
import json
from pathlib import Path


def probe(torch):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA_UNAVAILABLE: cannot validate GPU training on this runtime')
    device = torch.device('cuda:0')
    a = torch.arange(128 * 128, device=device, dtype=torch.float32).reshape(128, 128) / 16384
    output = a @ a.T
    torch.cuda.synchronize()
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError('GPU_NONFINITE_OUTPUT')
    return {'status':'PARTIAL','gate':'CUDA_MATMUL_PASS',
            'gpu':str(torch.cuda.get_device_name(0)),
            'torch_version':str(torch.__version__),
            'reconstruction_quality':'UNVERIFIED',
            'colmap_data_ingestion':'UNVERIFIED',
            'native_gsplat_training':'UNVERIFIED',
            'session_persistence':'NOT_PROVIDED_BY_KAGGLE'}


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='/kaggle/working/splatforge-gpu-probe.json')
    a=p.parse_args(argv)
    import torch
    result=probe(torch)
    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
