"""Mandatory native end-to-end test; intentionally fails without real packages."""
from fastapi.testclient import TestClient
from native_bridge.app import app
import hashlib


def test_both_physics_engines_and_cpu_splat_run_in_one_request():
    c = TestClient(app)
    response = c.post('/v1/e2e', json={
        'name': 'ASTRA SOULBRIDGE Ω',
        'memories': ['Premier environnement MuJoCo', 'Transfert vers PyBullet'],
    })
    assert response.status_code == 200
    proof = response.json()
    assert proof['status'] == 'PASS'
    assert proof['capsule_transfer_verified']
    assert proof['source_A']['engine'] == 'MuJoCo'
    assert proof['source_A']['pass'] and proof['source_A']['contact']
    assert proof['target_B']['engine'] == 'PyBullet'
    assert proof['target_B']['pass'] and proof['target_B']['contact']
    assert proof['splat']['pass'] and proof['splat']['gaussians'] == 3
    image = c.get('/v1/splat/sample.png')
    assert image.status_code == 200
    assert image.content.startswith(b'\x89PNG\r\n\x1a\n')
    assert image.headers['x-astra-proof-sha256'] == hashlib.sha256(image.content).hexdigest()
    assert image.headers['x-astra-proof-sha256'] == proof['splat']['png_sha256']