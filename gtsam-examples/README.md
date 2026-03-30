# Dataset Directory

This directory contains the datasets used in the 2026 FnT Article

"Smoothing Out the Edges:  Continuous-Time Estimation with Gaussian Process Motion Priors on Factor Graphs"

# Datasets


# Build Instructions

Make sure that GTSAM has been built prior to building the examples in this directory. Our specific version of GTSAM (with WNOA support) can be found in the `gtsam` directory. Follow the standard build instructions for GTSAM in that directory.

These examples also make use of `yaml-cpp` provided in the `extern` directory. This library must also be built prior to building examples.

Run this snippet to build these examples:
```bash
mkdir build \
&& cd build \
&& cmake .. \
&& make -j
```
Example runs should then appear in the `build` directory. Each example also makes use of .yaml files to load parameters.

# Additional Details

Additional information about these datasets (e.g. data collection, labelling, etc.) can be found in [AssignmentsDatasetDescription.pdf](AssignmentsDatasetDescription.pdf). 