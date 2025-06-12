# Neutrino-Simulator

**Quantum Simulation of Two-Flavour Neutrino Oscillations using Qiskit**  
*Author: Joseph Beeks*

---

## Overview

This repository accompanies my final year Master's research project, and presents a quantum computational framework to simulate neutrino flavour oscillations in the simplified two-flavour vacuum oscillation scenario. The simulation is implemented using IBM’s [Qiskit](https://qiskit.org) and benchmarked against classical analytical results derived from standard neutrino oscillation theory.

The code is structured to facilitate simulation on both real quantum hardware and fake-backends. High statistical fidelity is achieved through repeated sampling and error analysis, and the simulator has been validated through classical comparison with theoretical expectations.

---

## Usage

This requires you have an IBM token to run on real quantum hardware which you can put in the config.py. 