This repo contains latex and coded examples for a 2026 FnT Article:

"Smoothing Out the Edges:  Continuous-Time Estimation with Gaussian Process Motion Priors on Factor Graphs"

latex/
-contains the latex code for the manuscript

python/
-contains python code for simple examples

gtsam-analyses/
- contains analysis code for GTSAM examples and parameter files for the figures and analysis in the FnT article
- Follow README in that folder for instructions on how to run the analyses

gtsam-examples/
- Contains GTSAM code and data for the three sample datasets that were introduced in the FnT article.
- Follow the instructions in the README in that folder to run the examples

# Dev Container setup

If you want to work inside the container, use VS Code to run the devcontainer (it should start automatically). Make sure that both your UID and GID are set by default (otherwise default values will be used and may cause permission issues).

# Setup without Dev Container

If you don't want to use the dev container, then you must ensure that all of the required libraries are installed. Please see `.devcontainer/Dockerfile` for details.

If you want to run the steam regression test (`gtsam_analyses/steam-regression`)

# GTSAM Setup

To run the examples in the `gtsam-analyses` or `gtsam-examples` directory, you must first build GTSAM and make sure that it is available to be found by `cmake` in the analyses folder. First clone the GTSAM repository into the `extern` folder:
```bash
cd extern \
&& git clone https://github.com/borglab/gtsam.git \
```
Once the repository is cloned, you can build GTSAM using the following commands from within the `extern/gtsam` directory:
```bash
&& mkdir build \
&& cd build \
&& cmake ..\ 
&& make -j8 
```
More detailed build instructions can be found at [the GTSAM build page](https://gtsam.org/build/). Once built, GTSAM should be available to be found by `cmake` in the analyses folder. You may need to set the `GTSAM_DIR` environment variable to point to the location of the GTSAM build directory, which contains the `GTSAMConfig.cmake` file. For example:
```bash
export GTSAM_DIR=/path/to/gtsam/build
```



