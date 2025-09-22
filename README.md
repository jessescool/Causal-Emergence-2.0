# An Implementation of Causal Emergence 2.0

I implement the computational tools suggested in Erik Hoel's novel [Causal Emergence 2.0](https://arxiv.org/abs/2503.13395). 

Causal Emergence 2.0 (CE 2.0) attempts to capture causally-important scales of discrete dynamical systems, leveraging [first-principles notions of causality](https://arxiv.org/abs/2202.01854).

## Overview

The CE 2.0 framework explores how macro-scale representations of a system can exhibit greater causal power than their micro-scale counterparts. This implementation provides tools to:
- Calculate Causal Emergence scores
- Visualize apportionment paths
- Explore merge operations and which causal component is contributory

## Core Components

### `ce.py`
Core implementation; components to generate a greedy causal apportionment path.

### `ce_systems.py`
Common/predefined system configurations and example transition probability matrices (TPMs) for testing and analysis.

## Analysis Notebooks

### `ce_explorer.ipynb`
The frontend, where you can ask about downstream causal apportionment by providing a microscale TPM.
- `explore()` function for comprehensive system analysis
- Visualization of CP changes and absolute scores

### `det_spec_explorer.ipynb`
Getting some intuiton for how the greedy merge operates. When do we merge for determinism, when for specificity?
- Side-by-side matrix comparisons for apportionment paths under different metrics
- Path convergence analysis
- Status reporting for different metrics

## Usage

Import the core modules and explore a predefined system from the collection in `ce_systems.py`. The explore function generates and visualizes the complete greedy causal apportionment path for any given transition probability matrix. Examine how causal power changes through successive merges.

For comparative analysis between different merging metrics, use the `det_spec_explorer.py` to examine how determinism-only and specificity-only forced paths differ from a standard CE 2.0 apportionment. This intution is key in understanding CE 2.0 and underlying causal dynamics.

Provide your own transition probability matrix to the `explore()` function in `ce_explorer.py` to generate the apportionment sequence. The system expects $n \by n$ matrices representing the transition probabilities between all possible states of your discrete dynamical system microstate.

## Dependencies

- NumPy (matrix operations and numerical computing)
- SciPy (entropy calculations)
- Matplotlib (visualization)
- NetworkX (graph operations in `ce_systems.py`)

## Other

Among other systems, I looked at [Boid](https://en.wikipedia.org/wiki/Boids) behavior. Some demonstrations are included in `boids_under_CE.zip`.