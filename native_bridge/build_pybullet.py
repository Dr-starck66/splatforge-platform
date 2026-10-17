"""Install the verified conda-forge CPython 3.12 Bullet extension on Vercel.

PyPI pybullet 3.2.7 offers no Python 3.12 manylinux wheel. This script
recovers a matching open-source conda-forge binary with SHA-256 verification.
No install happens on user devices; only within the serverless build.
"""
import hashlib
import io
import os
import pathlib
import platform
import sys
import tarfile
import urllib.request
import zipfile

import zstandard

NAME = "pybullet-3.25-py312hf49885f_5.conda"
URL = f"https://conda.anaconda.org/conda-forge/linux-64/{NAME}"
SHA256 = "849bbe715c3d3e3c89f19a096d0158ce712022f387829ba222c327c533b747d4"
TARGET = pathlib.Path(__file__).resolve().parent


def main():
    if sys.version_info[:2] != (3, 12) or platform.machine() != "x86_64":
        raise RuntimeError(f"Only CPython 3.12 x86_64 supported, not {sys.version} {platform.machine()}")
    data = urllib.request.urlopen(URL, timeout=70).read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != SHA256:
        raise RuntimeError(f"Conda binary hash mismatch: {actual}")
    extracted = False
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        entry = next(v for v in z.namelist() if v.startswith("pkg-") and v.endswith(".tar.zst"))
        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(z.read(entry))) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                for member in archive:
                    name = pathlib.Path(member.name).name
                    if name == "pybullet.cpython-312-x86_64-linux-gnu.so":
                        source = archive.extractfile(member)
                        if source is None or member.size < 100000:
                            raise RuntimeError("Bullet native extension invalid")
                        out = TARGET / name
                        with out.open("wb") as fh:
                            while True:
                                chunk = source.read(1024 * 1024)
                                if not chunk:
                                    break
                                fh.write(chunk)
                        print("ASTRA_NATIVE_PYBULLET_BUILT", str(out), out.stat().st_size, "source_sha256", SHA256)
                        extracted = True
                        break
    if not extracted:
        raise RuntimeError("pybullet native extension absent from verified conda package")


if __name__ == "__main__":
    main()