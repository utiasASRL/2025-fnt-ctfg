This repo contains latex and coded examples for a 2025 FnT Article

"Smoothing Out the Edges:  Continuous-Time Estimation with Gaussian Process Motion Priors on Factor Graphs"

latex/
-contains the latex code for the manuscript

python/
-contain code for examples

gtsam_analyses/
- contains analysis code for GTSAM examples

gtsam-ct-factors/
- temporary submodule containing GTSAM CT Factor fork of GTSAM

# Dev Container setup

If you want to work inside the container, use VS Code to run the devcontainer (it should start automatically). Make sure that both your UID and GID are set by default (otherwise default values will be used and may cause permission issues).

# Setup without Dev Container

If you don't want to use the dev container, then you must ensure that all of the required libraries are installed. Please see `.devcontainer/Dockerfile` for details.

If you want to run the steam regression test (`gtsam_analyses/steam-regression`)

# GTSAM Setup

To run the examples in the `gtsam_analyses` directory, you must first build GTSAM and make sure that it is available to be found by `cmake` in the analyses folder. Use the following script to do so:
```bash
cd gtsam-ct-factors \
&& mkdir build \
&& cd build \
&& cmake ..\ 
&& make -j 
```
Note that this is temporary until our changes are merged properly into GTSAM.
