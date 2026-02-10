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

# Running Analyses

To run the examples in the `gtsam_analyses` directory, you must first build GTSAM and make sure that it is available to be found by `cmake` in the analyses folder. Use the following script to do so:

```bash
cd gtsam-ct-factors \
&& mkdir build \
&& cd build \
```