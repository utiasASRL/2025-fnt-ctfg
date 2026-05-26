# Dataset Directory

This directory contains the datasets used in the 2026 FnT Article

"Smoothing Out the Edges:  Continuous-Time Estimation with Gaussian Process Motion Priors on Factor Graphs"

# Datasets


# Build Instructions

## GTSAM Setup

Make sure that GTSAM has been built prior to building the examples in this directory. See top level README for instructions on how to build GTSAM.

## YAML-CPP Setup

These examples also make use of `yaml-cpp` provided in the `extern` directory. This library must also be built prior to building examples. Follow the instructions in `extern/yaml-cpp/install.txt` to build this library. 

# Build Examples

Run this snippet to build these examples:
```bash
mkdir build \
&& cd build \
&& cmake .. \
&& make -j
```
Example runs should then appear in the `build` directory. Each example also makes use of .yaml files to load parameters.

# Examples

To run the examples in this directory, use the following commands from the `gtsam-examples` directory:

| Name | Manifold | Run Instructions |
|------|----------|------------------|
| Giant Glass of Milk | R¹ | `./build/GiantGlassOfMilk` |
| Lost in the Woods | SE(2) | `./build/LostInTheWoods` |
| Starry Night | SE(3) | `./build/StarryNight` |

By default, the outputs will be saved to the `results` directory in the subfolder corresponding to each example. The parameters of each example can be modified by changing the corresponding .yaml file in each directory. Alternate config files can be passed by command line argument to the Lost in the Woods and Starry Night examples (the Giant Glass of Milk example only has one config file). For example, to run the Starry Night example with the `starryNightWNOA` config file, use the following command:
```bash
./build/StarryNight --config-file StarryNight/config/starryNightWNOA.yaml
```

Additional information about these datasets (e.g. data collection, labelling, etc.) can be found in [AssignmentsDatasetDescription.pdf](AssignmentsDatasetDescription.pdf). 

