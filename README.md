# Cognitive Robotics Final Assignment

## Project Overview
This project explores the implementation of the **CrossFormer Cross-Embodiment Policy** within a **PyBullet simulation environment**. The goal was to integrate the CrossFormer policy to enable cross-embodiment learning in robotic simulations. Due to time limitations, the policy was not fully implemented in the actual simulation. However, testing can be performed using the provided Jupyter Notebook.

## Setup Instructions

### 1. Create a Conda Environment
To ensure a clean and reproducible setup, create a new Conda environment:
```sh
conda create -n crossformer python=3.10
conda activate crossformer
```

### 2. Install Dependencies
Once inside the `crossformer` environment, install the required dependencies:
```sh
pip install -r requirements.txt
```

### 3. Install CrossFormer
Follow the official installation instructions to install CrossFormer:
- Repository: [CrossFormer](https://github.com/rail-berkeley/crossformer)
- Clone the repository and follow the installation steps provided in the README.

### 4. Install Airobot
Follow the official installation instructions to install Airobot:
- Repository: [Airobot](https://github.com/Improbable-AI/airobot)
- Clone the repository and follow the installation steps provided in the README.

## Testing the CrossFormer Policy
Due to time constraints, the full implementation of the CrossFormer policy in the simulation environment was not completed. However, to test its functionality, use the provided Jupyter Notebook:
```sh
jupyter notebook crossformer_playground.ipynb
```
This notebook allows experimentation with the CrossFormer policy without full simulation integration.

## Notes
- Ensure that all dependencies are correctly installed before running the notebook.
- Contributions or future extensions could involve fully integrating the policy into the simulation.
