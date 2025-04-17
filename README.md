# OpenAI-s-Gym-Cartpole-v0

This project demonstrates the optimization of the CartPole environment using a neural network trained with a genetic algorithm (GA). The goal is to balance the pole on the cart for as long as possible by evolving the neural network's weights and biases over multiple generations.

## Features

- **Neural Network**: A simple feedforward neural network with ReLU and softmax activation functions.
- **Genetic Algorithm**: Implements crossover, mutation, and selection to evolve the neural network's weights and biases.
- **CartPole Environment**: Uses OpenAI Gym's `CartPole-v0` environment for testing and training.
- **Visualization**: Plots the fitness scores over generations to visualize the optimization process.

## Requirements

- Python 3.7+
- Required Python libraries:
  - `gym`
  - `numpy`
  - `matplotlib`
  - `pyglet`

 Install the dependencies using pip:

```bash
pip install gym[all] numpy matplotlib pyglet
```
```bash
pip install gym==0.24.1
```

## How It Works
### Neural Network Initialization:

- The neural network is initialized with random weights and biases.
- The network predicts actions for the CartPole environment.
### Genetic Algorithm:

- A population of neural networks is evolved over multiple generations.
- The top-performing networks are selected for crossover and mutation to create the next generation.
### Training:

- The GA optimizes the weights and biases of the neural network to maximize the time the pole remains balanced.
### Testing:

- The best-performing neural network is tested in the CartPole environment.

## Usage
- Clone the repository:
  ```bash
  git clone https://github.com/Dheeraj-yellapu/OpenAI-s-Gym-Cartpole-v0
  ```
- Run the script:
  ```bash
  python3 cartpole.py
  ```
- The script will:
  - Train the neural network using the genetic algorithm.
  - Display the fitness graph over generations.
  - Test the best-performing neural network in the CartPole environment.
## Notes
- The CartPole environment will render during the testing phase. Close the rendering window to end the test.
- Adjust the hyperparameters (e.g., population size, number of generations, mutation rate) in the trainer and GA classes for experimentation.
