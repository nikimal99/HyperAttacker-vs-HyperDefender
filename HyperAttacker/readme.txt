# Hyperbolic Graph Attack Simulation

This folder provides an implementation for simulating hyperbolic attack. The attack process involves estimating Poincaré embeddings and using them as inputs to `HyperAttacker`.

## Setup Instructions

### 1. Clone the Repository

```sh
git clone https://github.com/nikimal99/HyperAttacker-vs-HyperDefender.git
cd HyperAttacker

### 2. Create and Activate the Conda Environment

conda env create -f environment.yml
conda activate hie_attack

### 3. Estimate Poincaré Embeddings
Before running the attack, you need to compute Poincaré embeddings. 

###4. Run the Hyperbolic Attack Simulation
Use the estimated embeddings and other necessary data as input to HyperAttacker:
python main_hyperattacker.py
