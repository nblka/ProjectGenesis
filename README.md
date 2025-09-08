# Project Genesis: A Computational Framework for Emergent Spacetime and Physics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/)

**Project Genesis is a simulation framework designed to explore a fundamental hypothesis: that Spacetime, Quantum Mechanics, and all physical laws are emergent properties of a deeper, purely relational, computational structure.**

This is not a simulation of physics *within* spacetime. It is an attempt to simulate the **genesis of spacetime itself**. We start with an abstract, pre-geometric entity—a graph or simplicial complex—defined only by its connectivity (topology). We then apply simple, local rules of interaction and observe if a structure resembling our 4D relativistic universe can emerge from these minimal ingredients.

Our core investigation revolves around a **co-evolutionary principle**: what if the 'quantum field' (information encoded on the graph) and the 'geometry' (the graph's topology) are not separate entities, but dynamically influence each other at the most fundamental level? This project aims to test if this principle is the key to the emergence of a stable, Lorentz-invariant reality.

---

## The Core Hypothesis in a Nutshell

We do not presuppose the existence of space. We begin with a **simplicial complex**—a network of nodes, edges, triangles, etc.—that represents the fundamental structure of reality. It has no inherent geometry, coordinates, or metric. The only defined property is **adjacency**: which nodes are connected to which.

On this structure, we define two co-dependent dynamics:

1. **"Matter" Dynamics:** A `psi` field (a complex number, representing a quantum phase) is defined on the nodes. It evolves locally, propagating from a node to its neighbors.
2. **"Spacetime" Dynamics:** The topology of the complex itself is not static. It can locally reconfigure through **topological transformations** (e.g., Pachner moves). Critically, the rule governing these transformations depends on the local state of the `psi` field. *(not yet implemented)*

Our primary goal is to find a set of rules for this co-evolution that results in a system which, on a large scale, appears to be a smooth, continuous spacetime of a specific dimension (e.g., 3+1) and whose excitations (stable, propagating patterns in `psi` and the topology) behave like relativistic quantum particles.

---

## Project Structure

The project is designed to be modular and scalable, separating the core stages of an experiment into distinct scripts.

- `topologies.py`: Defines the initial topological structure of the universe. The current `CrystalTopology` implements a highly-ordered 2D triangular complex, which serves as a simple, 'low-entropy' starting point for our evolutionary simulations. The 2D coordinates associated with the nodes are used strictly for visualization purposes and do not participate in the physics of the simulation itself.
- `initial_conditions.py`: Sets up the "Big Bang" for the simulation. Defines states like a chaotic `PrimordialSoupState` or a structured `VortexState`.
- `simulation.py`: The core simulation engine. It evolves the `psi` field step-by-step.
- `tracker.py`: The "observer". An intelligent analysis module that scans the universe at each step to detect, characterize, and track emergent quasi-particles based on their quantum numbers (mass, charge, etc.).
- `main.py`: The main executable to launch a simulation run. It initializes everything, runs the simulation loop, and saves the raw state of the universe at each frame to `.npz` files.
- `renderer.py` / `render_frames.py`: The visualization pipeline. `render_frames.py` takes the raw data from a run and uses `renderer.py` in a massively parallel fashion to create beautiful high-resolution PNG frames.
- `compile_video.py`: The final stage. It uses `ffmpeg` to compile the rendered PNG frames into a professional, high-quality MP4 video.
- `test_renderer.py`: A utility script to render a single, specific frame from a run, perfect for debugging and analysis.

---

## Getting Started

### Prerequisites

- Python 3.9+
- `ffmpeg`: You must have ffmpeg installed and available in your system's PATH.
- Required Python packages:
  ```bash
  pip install numpy scipy matplotlib tqdm termcolor networkx
  ```

### Running an Experiment (The 3-Step Workflow)

**Step 1: Run the Simulation**

This is the main computational step. You create a universe and let it evolve, saving the data.

```bash
# Example: Run a simulation with a specific seed, on a 80x60 grid,
# starting from a 'vortex' initial condition, for 3000 frames.
python main.py --seed 12345 --width 80 --height 60 --ic vortex --frames 3000
```

This will create a new directory, e.g., `run_SEED_12345_crystal_80x60_vortex/`, containing a `data/` subfolder with all the `.npz` frame data and a `metadata.json` file.

**Step 2: Render the Frames**

Now, turn the raw data into images.

```bash
# Render every 5th frame from the run directory we just created
python render_frames.py run_SEED_12345_crystal_80x60_vortex/ -rs 5
```

This will create a `frames/` subfolder inside the run directory, filled with PNG images. Rendering is parallelized and will use all available CPU cores.

**Step 3: Compile the Video**

Finally, assemble the frames into a movie.

```bash
# Compile the video at 30 frames per second
python compile_video.py run_SEED_12345_crystal_80x60_vortex/ -fr 30
```

This will create a final `genesis_..._v10.5.mp4` file in the run directory. The temporary `frames/` directory will be deleted unless you specify the `--keep-frames` flag.

---

## Future Research & How to Contribute

This framework is the starting point for a vast research program. Key future directions include:

- **Implementing Dynamic Topology:** Adding the "edge flip" mechanism to `simulation.py` and making it dependent on `psi`. This is the highest priority.
- **Exploring the Rule Space:** Systematically testing different `U_R` rules to see which ones produce stable, relativistic particles.
- **3D and 4D Topologies:** Extending `topologies.py` to support 3D and 4D simplicial complexes.
- **Analyzing Interactions:** Creating initial conditions with multiple vortices to study how they interact, scatter, and potentially annihilate.

We welcome contributions from physicists, mathematicians, and computer scientists. If you have an idea, feel free to open an issue to discuss it or submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License.
