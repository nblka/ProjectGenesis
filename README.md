# Project Genesis: Breathing Causality

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/)

**Project Genesis is a simulation framework designed to explore a fundamental hypothesis: that Quantum Mechanics and the Arrow of Time are emergent properties of a dynamic causal network built upon a static, pre-geometric substrate.**

This is not a simulation of physics *on* a graph. It is an attempt to simulate the **genesis of physical law itself**. We start with a static, undirected graph—the **Substrate**—representing a timeless web of potential relationships. On this substrate, we introduce two co-dependent dynamics:

1.  A **quantum field (`psi`)**, which evolves locally, propagating its phase and amplitude across the nodes of the substrate.
2.  A **dynamic causal graph (`G_causal`)**, whose directed edges (`A → B`) are determined at every instant by the local state of the `psi` field.

Our core investigation revolves around a **co-evolutionary principle**: Matter tells Causality which way to flow, and Causality tells Matter how to evolve. We test if this "Breathing Causality" is the key to the emergence of a universe that looks quantum-mechanical and possesses a definite Arrow of Time.

---

## The Core Hypothesis in a Nutshell

We begin with a **static, undirected graph** (the Substrate), defined only by its connectivity. This represents a timeless, pre-causal set of all possible adjacency relations. It has no inherent geometry or directionality.

The simulation loop then follows a simple, profound cycle:

1.  **`psi(t) → G_causal(t)`:** At time `t`, the current state of the quantum field `psi` determines the instantaneous **directed causal graph**. The edges of the substrate are given a direction based on the local gradient of `psi`. For example, a causal link `A → B` forms if the amplitude at `A` is higher than at `B` (or vice versa, depending on the chosen hypothesis).
2.  **`G_causal(t) → psi(t+1)`:** The quantum field `psi` then evolves for one step, but it can **only propagate along the directed edges** of the causal graph `G_causal(t)` that was just formed.

This creates a feedback loop. Regions of high amplitude 'project' a causal structure onto their neighborhood, and this very structure then dictates how that amplitude will spread or concentrate in the next moment. Our primary goal is to see if stable, particle-like patterns emerge from this loop and if their collective behavior matches the predictions of Quantum Field Theory.

---

## Project Structure

The project is designed to be modular, allowing for easy swapping of different physical hypotheses.

- **`topologies.py`:** Defines the **static, undirected Substrate**. The current `generate_crystal_topology` creates a highly-ordered 2D triangular lattice, serving as a simple, 'low-entropy' background. The 2D coordinates are used strictly for visualization.
- **`initial_conditions.py`:** Sets the initial state of the `psi` field on the substrate. Defines states like a chaotic `PrimordialSoupState` or a structured `VortexState`.
- **`causality.py`:** **The heart of the new hypothesis.** This module contains different strategies for the `psi → G_causal` step. It defines the rules by which the quantum field directs the flow of causality (e.g., `AmplitudeConvergentCausality`).
- **`simulation.py`:** The core engine. It orchestrates the "Breathing Causality" loop: calls `causality.py` to determine the directed graph for the current step, and then evolves `psi` along that graph.
- **`tracker.py`:** The "observer". An analysis module that scans the `psi` field to detect and track emergent, stable quasi-particles.
- **`analytics.py`:** Gathers global statistics from the run, such as the evolution of entropy and a quantitative measure of the **"Arrow of Time"** (by measuring the asymmetry of the causal flow).
- **`main.py`:** The main executable to configure and launch a simulation run. It saves the full state of the universe (`psi` and `G_causal`) at each frame.
- **`renderer_worker.py` / `render_frames.py`:** The parallelized visualization pipeline for creating PNG frames from the raw data.
- **`compile_video.py`:** The final stage, compiling frames into a high-quality MP4 video using `ffmpeg`.
- **`utils.py`, `test_*.py`:** Utility functions and a comprehensive suite of unit tests to ensure the robustness of each component.

---

## Getting Started

### Prerequisites

- Python 3.9+
- `ffmpeg`: Must be installed and available in your system's PATH.
- Required Python packages:
  ```bash
  pip install numpy scipy matplotlib tqdm termcolor
  ```

### Running an Experiment (The 3-Step Workflow)

**Step 1: Run the Simulation**
This step generates all the raw data for your universe's history.

```bash
# Example: Run a simulation with a seed, on a 80x60 grid,
# starting from a 'vortex', using the 'convergent' causality rule.
python main.py --seed 12345 --width 80 --height 60 --ic vortex --causality convergent --frames 3000
```
This creates a directory like `run_SEED_12345_crystal80x60_vortex_convergent/` containing all raw `.npz` data and metadata.

**Step 2: Render the Frames**
Turn the raw data into images.

```bash
# Render every 5th frame from the run directory we just created
python render_frames.py run_SEED_12345_crystal80x60_vortex_convergent/ -rs 5
```
This creates a `frames/` subfolder inside the run directory.

**Step 3: Compile the Video**
Assemble the frames into a movie.

```bash
# Compile the video at 30 frames per second
python compile_video.py run_SEED_12345_crystal80x60_vortex_convergent/ -fr 30
```
This creates the final `.mp4` video file in the run directory.

---

## Future Research & How to Contribute

This framework is the starting point for a vast research program. Key future directions include:

- **Exploring the Rule Space:** Systematically testing different causality rules in `causality.py`. How does the choice of rule affect particle stability and interactions?
- **3D Substrates:** Extending `topologies.py` to support 3D static substrates (e.g., a cubic lattice or a random 3D graph).
- **Multi-Particle Systems:** Creating initial conditions with multiple vortices to study emergent interactions, scattering, and annihilation.
- **Metric Emergence:** Developing new analytics to measure the effective spacetime metric that arises from the dynamic causal graph (e.g., by measuring signal propagation times).

We welcome contributions. If you have an idea, feel free to open an issue to discuss it or submit a pull request.

## License

This project is licensed under the MIT License.
