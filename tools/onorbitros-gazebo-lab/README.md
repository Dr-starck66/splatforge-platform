# OnOrbitROS noetic / Gazebo public validation runner (isolated branch)

This is a minimal, **public**, generic telemetry probe. No private ASTRA Ω code is copied into this repository. It builds the actual **OnOrbitROS/Simulation** `orbit_publisher_pkg` C++ node from pinned commit `c2edc7414cc90a6f0421c16702ef44c5895fd151`, with a build metadata fix removing unused `orbit_msgs` dependency; the upstream algorithm and MIT license are unchanged.

Docker starts ROS 1 Noetic, `gazebo_ros empty_world` (headless, real `/clock`), and upstream's dynamic orbit publisher; the Python web service subscribes read-only to `/OrbitPosition`, `/OrbitVelocity` and `/clock` and returns a JSON snapshot at `/evidence` or `/frame`. **There are no robot actuators or commands**, and this does not model a truss or orbital structure.

To consider the ROS integration proven, `/evidence` must report `status=LIVE_ROS_GAZEBO_OBSERVED`, topic counters >0, source node and Gazebo node detected, plausible finite arrays and quaternion norm, zero robot command count. A 200 HTTP alone is not proof.

Powered by legacy ROS Noetic (EOL May 2025), **research isolation only**. Render Free is constrained to 0.1 vCPU, 512MB RAM, ephemeral storage and sleeps when idle; the simulation may fail due to RAM or legacy dependency issues. No paid resources requested.

Run locally if Docker is available:

```
docker build -t onorbitros-probe .
docker run --rm -p 10000:10000 -e PORT=10000 onorbitros-probe
curl -s http://localhost:10000/evidence
```

Upstream source: https://github.com/OnOrbitROS/Simulation (MIT). No relation to Caltech ASTRA.
