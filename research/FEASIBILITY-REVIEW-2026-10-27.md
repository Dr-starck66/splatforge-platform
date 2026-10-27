# SplatForge — Independent scientific feasibility review (not a research discovery)
**27 October 2026 | Status: PARTIAL | No additional expenditure | External opinion requested, not yet obtained**

## Current project and proof boundary

Public source: [Dr-starck66/splatforge-platform](https://github.com/Dr-starck66/splatforge-platform). The current code implements an **experimental CPU-bound optimization of a small Gaussian scene**, requiring COLMAP camera poses, sparse 3D points, and RGB input. It exports a binary `.splat` representation using a bounded isotropic Gaussian forward-rendering approximation. It is **not an implementation equivalent to full CUDA-native 3D Gaussian Splatting**.

The README records a prior local synthetic-fixture experiment with **five synthetic images, 12 points, 80 optimization steps, MSE from 0.0073640458 to 0.0000105124, and a 384-byte `.splat` export**. This is reported historical evidence and was **not independently rerun in this review**. The `tests/test_cpu_colmap_trainer.py` test creates synthetic 32×32 views. The current trainer `tools/train_colmap_cpu.py` measures initial and final MSE **over the same views used for training**, not held-out photographs. Thus the MSE decrease validates neither realistic scene quality, generalization nor superiority to prior methods. No publicly verified real-photo 3D reconstruction, CPU-versus-GPU benchmark, independent laboratory replication or hosted end-to-end demonstration is established.

## Proposed falsifiable engineering-science question

**Can an entirely CPU-bounded Gaussian-optimization path produce useful reconstructions and unseen-view renders on real multi-view photographs at low memory/compute cost, compared with a simple CPU point-reprojection baseline, while documenting the expected quality tradeoff against established GPU 3DGS?**

This is a **question about engineering feasibility and quantifiable performance**, **not a claim of new 3DGS invention**. Original 3DGS methods and major improvements come from Inria GRAPHDECO and other established research groups; any algorithmic novelty requires a separate review against published work and licenses.

## Proposed independent evaluation — preregister before first real benchmark

1. **Input:** at least three documented real-photo static scenes from appropriately licensed benchmark sets or authorized photographs. Record acquisition, resolution, camera calibration, hold-out policy, license, per-scene file SHA-256, hardware CPU/threads/RAM, Python packages and code SHA.
2. **Split:** reserve >=20% camera views for inference-only **before optimization**; no view or photometric-derived target from the test subset enters training or parameter selection; distinct scene-level holdout for future tuning.
3. **Baselines:** same-pose nearest-point color reprojection or another reproducible CPU baseline; report established GPU 3DGS from published references only as external context unless it is actually run on equivalent scenes/hardware. Label non-comparable resources explicitly.
4. **Primary outcomes:** unseen-view **PSNR and SSIM** by scene; record scene latency, total CPU wall-clock, peak RSS, artifact size and reconstruction failure rates. PSNR/SSIM target thresholds must be set based on a pilot **before evaluating held-out results**; absent such preregistration, exploratory analysis cannot count as a confirmatory success.
5. **Negative controls / ablations:** shuffled camera poses, broken calibration, missing point geometry, unchanged initial model, correctly handled zero-view or out-of-memory cases, and fixed-seed repeatability. Report failures and all scenes, not just successes.
6. **Acceptance:** no `PASS_SCIENTIFIC` until meaningful unseen-view performance and resource claims are evidenced on **real** photographs with reproducible scripts, scene holdouts, at least one credible baseline, intact negative controls and critical independent review. No automatic declaration of superiority or originality.

## Two potential independent scientific contacts

1. **Inria GRAPHDECO — George Drettakis** (team leader, specialist in 3D Gaussian Splatting and novel-view synthesis). Official institutional contact: **George.Drettakis@inria.fr** listed on [French government research-unit directory](https://appliweb.dgri.education.fr/rnsr/PresenteStruct.jsp?PUBLIC=OK&numNatStruct=201521163T), and [Inria research positions](https://jobs.inria.fr/public/classic/en/offres/2026-10425). **Request**: a short no-obligation critical opinion on the test protocol and whether CPU-only reconstruction provides a useful research question. This is not a request for free experimental work or an endorsed partnership.
2. **Inria COMPACT, Rennes — Christine Guillemot** (researcher on compact and efficient Gaussian Splatting, including CAFe-GS): [2025 team report](https://radar.inria.fr/report/2025/compact/index.html). Her institutional email is named in [an Inria job description](https://jobs.inria.fr/public/classic/en/offres/2026-09913/topdf) via `firstname.lastname@inria.fr`; an older Inria document explicitly lists `Christine.Guillemot@inria.fr`. Independently confirm that the mailbox is still current before any direct outreach. **Request**: evaluation-design feedback (resource/quality tradeoff), not an endorsement.

**Email safety and ethics:** the institutional listings establish publicly published addresses, not proof of inbox deliverability or consent to unsolicited campaigns. Check prior sent history, failures and refusals before sending. Only a targeted, personalized scientific feasibility inquiry after review of the text, not a bulk email. **No message was sent during this review.**

## Ready-to-use brief for an expert

SplatForge is an open-source experimental CPU-based Gaussian scene optimization and export pipeline. Its first synthetic-scene checks indicate a working optimization path, but training loss is measured on training views; no real-scene or unseen-view performance has yet been demonstrated. We would appreciate a short critical assessment of whether a GPU-free, carefully resource-constrained baseline evaluation on real multi-view photographs is methodologically meaningful and what negative controls and quality/cost baselines you would recommend. We do not claim algorithmic novelty, published scientific validation, full 3DGS fidelity or a committed collaboration.

## Status summary

- **PASS (documentation):** experiment plan, boundaries and institutional source links recorded in GitHub.
- **PARTIAL (implementation):** code and historically reported synthetic test proof; actual re-execution and real-photo generalization unverified.
- **UNVERIFIED (scientific):** real scene PSNR/SSIM, performance vs. baselines, uniqueness of method, independent acceptance, deployed end-to-end service.
- **NOT SENT (outreach):** prospective contacts only.
