# Connect4 Deep Q-Learning Model
Reinforcement learning model for Connect 4.

## Reinforcement Learning
This project utilizes Deep Q-Learning, a reinforcement learning technique where the agent optimizes its actions by approximating the Q-value function. The Deel Learning Model is trained to predict the long-term reward for each possible action based on the current state of the board.

## Model Design
### Layers
This model consists of two convolutional layers, and two fully connected layers:
- Convolutional layers are responsible for capturing local features of the game board.
- Fully connected layers will process those features and convert them into Q-value predictions.

### Optimizations
- Replay Memory: Stores a buffer of past transitions (state, action, reward, next state) for batch training, significantly improving training stability.
- Clipped Double Q-Learning: Addresses overestimation of Q-values by separately evaluating and selecting actions using two q-networks [Original paper](https://arxiv.org/pdf/1509.06461).
- Exploration Decay: Gradually reduces epsilon to shift from exploration (random actions) to exploitation (learned strategies).

### Training Process
- Self-play: The agent is trained on a randomly initialized adversary model for a given number of episodes to develop strategies while preventing overfitting towards specific models.
- Human-training: Optionally, the agent can learn by playing against human players to diversify strategies.
- Random-training: Initial training against random actions to bootstrap learning and avoid overfitting to a specific opponent.

## Future optimizations
- Strategic reward system: the current model struggles to find defensive moves. Therefore, a reward system needs to be set up to encourage defensive actions.
