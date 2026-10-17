from native_bridge.app import _cpu_splats, digest, health


def test_gaussian_pixels():
    raw, info = _cpu_splats()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert info["pass"] and info["png_bytes"] == len(raw)


def test_digest_order_invariant():
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})


def test_health_does_not_claim_cuda():
    assert health()["cuda_gsplat"] == "NOT_TESTED"