"""ASTRA Native Engines Smoke v1 — MuJoCo, PyBullet and gsplat CUDA.
Requires explicit no-cost compatible runner (e.g. Kaggle free T4x2 notebook).
Do not report PASS unless each actual native test runs successfully.
"""
from __future__ import annotations
import datetime
import hashlib
import json
import sys
from pathlib import Path

def main():
    proof = {
        'schema': 'astra-native-smoke/1',
        'executed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'runtime': sys.version,
        'engines': {},
        'scope': 'Native Mujoco/PyBullet physics and gsplat CUDA autograd smoke; NOT full SOULBRIDGE or real-photo SplatForge'
    }
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('astra_native_proof.json')
    try:
        import mujoco
        xml = '<mujoco model="freefall"><option timestep="0.005" gravity="0 0 -9.81"/><worldbody><geom type="plane" size="3 3 .1"/><body pos="0 0 2"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        z_before = float(data.qpos[2])
        for _ in range(500): mujoco.mj_step(model, data)
        z_after = float(data.qpos[2])
        assert abs(z_before - 2.0) < 1e-6 and 0.07 < z_after < 0.15, (z_before, z_after)
        proof['engines']['mujoco'] = {'state':'PASS','z_before':z_before,'z_after':z_after,'version':mujoco.__version__}
    except Exception as error:
        proof['engines']['mujoco'] = {'state':'FAIL','error':str(error)}
    try:
        import pybullet as p
        client = p.connect(p.DIRECT)
        try:
            p.setGravity(0,0,-9.81,physicsClientId=client)
            p.setTimeStep(1/240,physicsClientId=client)
            floor = p.createCollisionShape(p.GEOM_PLANE,physicsClientId=client)
            p.createMultiBody(baseMass=0,baseCollisionShapeIndex=floor,physicsClientId=client)
            sphere = p.createCollisionShape(p.GEOM_SPHERE,radius=.1,physicsClientId=client)
            body = p.createMultiBody(baseMass=1,baseCollisionShapeIndex=sphere,basePosition=[0,0,2],physicsClientId=client)
            before = p.getBasePositionAndOrientation(body,physicsClientId=client)[0][2]
            for _ in range(600): p.stepSimulation(physicsClientId=client)
            after = p.getBasePositionAndOrientation(body,physicsClientId=client)[0][2]
            assert before > 1.99 and .07 < after < .15, (before,after)
            proof['engines']['pybullet'] = {'state':'PASS','z_before':before,'z_after':after,'steps':600}
        finally: p.disconnect(client)
    except Exception as error:
        proof['engines']['pybullet'] = {'state':'FAIL','error':str(error)}
    try:
        import torch
        import gsplat
        if not torch.cuda.is_available(): raise RuntimeError('CUDA accelerator is not available')
        torch.manual_seed(20261017)
        d='cuda'
        means=torch.tensor([[0.,0.,2.],[.15,0.,2.3]],device=d)
        quats=torch.tensor([[1.,0.,0.,0.],[1.,0.,0.,0.]],device=d)
        scales=torch.tensor([[.22,.22,.22],[.2,.2,.2]],device=d)
        opacities=torch.tensor([.95,.9],device=d)
        viewmats=torch.eye(4,device=d).unsqueeze(0)
        Ks=torch.tensor([[[64.,0.,32.],[0.,64.,32.],[0.,0.,1.]]],device=d)
        def render(colors):
            return gsplat.rasterization(means,quats,scales,opacities,colors,viewmats,Ks,width=64,height=64)[0]
        true_colors=torch.tensor([[.9,.2,.1],[.1,.7,.2]],device=d)
        with torch.no_grad(): target=render(true_colors).detach()
        colors=torch.nn.Parameter(torch.full((2,3),.5,device=d))
        optim=torch.optim.Adam([colors],lr=.045)
        start=float(((render(colors)-target)**2).mean().item())
        for _ in range(35):
            optim.zero_grad()
            loss=((render(colors)-target)**2).mean()
            loss.backward()
            optim.step()
        final=float(((render(colors)-target)**2).mean().item())
        assert start>0 and final<start*.4 and torch.isfinite(colors).all(), (start,final)
        proof['engines']['gsplat_cuda'] = {'state':'PASS','gpu':torch.cuda.get_device_name(0),'steps':35,'loss_before':start,'loss_after':final}
    except Exception as error:
        proof['engines']['gsplat_cuda'] = {'state':'FAIL','error':str(error)}
    proof['status']='PASS_ALL_NATIVE' if all(
        proof['engines'].get(k,{}).get('state') == 'PASS'
        for k in ['mujoco','pybullet','gsplat_cuda']) else 'FAIL_NATIVE_GATES'
    output.write_text(json.dumps(proof, indent=2, ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps(proof, indent=2, ensure_ascii=False))
    print('Proof file:',output,'sha256:',hashlib.sha256(output.read_bytes()).hexdigest())
    return 0 if proof['status']=='PASS_ALL_NATIVE' else 1

if __name__=='__main__':
    raise SystemExit(main())
