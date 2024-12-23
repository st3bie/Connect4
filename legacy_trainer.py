import torch
import numpy as np

from game.Connect4 import Connect4
from engine.Model import DQNAgent

def human_vs_ai(env, agent, human_player=1):
    ai_player = -human_player

    # Initialize the game
    state = env.reset()
    env.render_board()
    done = False

    while not done:
        if env.current_player == human_player:
            # Human's turn
            valid_move = False
            while not valid_move:
                try:
                    action = int(input(f"Your turn! Choose a column (0-{env.cols - 1}): "))
                    if env.is_valid_action(action):
                        valid_move = True
                    else:
                        print("Invalid move! Column is full or out of range. Try again.")
                except ValueError:
                    print("Invalid input! Please enter an integer.")
            
            # Execute the human's move
            next_state, reward, done = env.drop(human_player, action)
            env.render_board()

            if done:
                if reward == 10.0:
                    print("Congratulations! You win!")
                elif reward == -20.0:
                    print("Invalid move! You lose!")
                else:
                    print("It's a tie!")
                break

        else:
            print("AI's turn...")
            action = agent.select_action(state)
            print(f"AI chooses column {action}")
            next_state, reward, done = env.drop(ai_player, action)
            env.render_board()

            if done:
                if reward == 10.0:
                    print("AI wins! Better luck next time.")
                elif reward == -20.0:
                    print("AI made an invalid move! You win!")
                else:
                    print("It's a tie!")
                break

        # Update the state
        state = next_state

def self_train(env, agent1, agent2, num_episodes):
    episode_rewards1 = []
    episode_rewards2 = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward1 = 0.0
        total_reward2 = 0.0
        turn = 1

        while not done:
            if turn % 2 == 1:
                current_player = 1
                state = env.get_state(current_player)
                action = agent1.select_action(state)

                last_state1 = state
                last_action1 = action

                next_state, reward, done = env.drop(current_player, action)

                agent1.store_transition(state, action, reward, next_state, float(done))

                if done:
                    if reward == 10.0:
                        agent1_reward = 10.0
                        agent2_reward = -10.0

                        if last_state2 is not None and last_action2 is not None:
                            agent2.store_transition(last_state2, last_action2, agent2_reward, next_state, float(done))

                    elif reward == -20.0:

                        agent1_reward = -20.0
                        agent2_reward = 0.0

                    else:
                        agent1_reward = 0.0
                        agent2_reward = 0.0

                        if last_state2 is not None and last_action2 is not None:
                            agent2.store_transition(last_state2, last_action2, agent2_reward, next_state, float(done))
                else:
                    agent1_reward = 0.0
                    agent2_reward = 0.0

                total_reward1 += agent1_reward
                total_reward2 += agent2_reward
                
            else:
                current_player = -1
                state = env.get_state(current_player)
                action = agent2.select_action(state)

                last_state2 = state
                last_action2 = action

                next_state, reward, done = env.drop(current_player, action)
                agent2.store_transition(state, action, reward, next_state, float(done))
                
                if done:
                    if reward == 10.0:
                        agent2_reward = 10.0
                        agent1_reward = -10.0

                        if last_state1 is not None and last_action1 is not None:
                            agent1.store_transition(last_state1, last_action1, agent1_reward, next_state, float(done))
                    
                    elif reward == -20.0:
                        agent2_reward = -20.0
                        agent1_reward = 0.0
                    else:
                        agent2_reward = 0.0
                        agent1_reward = 0.0

                        if last_state1 is not None and last_action1 is not None:
                            agent1.store_transition(last_state1, last_action1, agent1_reward, next_state, float(done))
                else:
                    agent2_reward = 0.0
                    agent1_reward = 0.0

                total_reward1 += agent1_reward
                total_reward2 += agent2_reward

            turn += 1

        agent1.optimize()
        agent2.optimize()
        episode_rewards1.append(total_reward1)
        episode_rewards2.append(total_reward2)
        
        if (episode + 1) % 1000 == 0:
            avg_r1 = np.mean(episode_rewards1[-1000:])
            avg_r2 = np.mean(episode_rewards2[-1000:])
            print(f"Model 1 Average Reward (Episode {episode-999}-{episode+1}): {avg_r1}")
            print(f"Model 2 Average Reward (Episode {episode-999}-{episode+1}): {avg_r2}")
        
        if (episode + 1)% 5000 == 0:
            agent1.save_model("model1.pth")
            agent2.save_model("model2.pth")

def diverse_train(env, agent1, episodes, num_agents):

    for i in range(num_agents):
        agent2 = DQNAgent(device=device)
        print("Training on agent: " + str(i))
        self_train(env, agent1, agent2, episodes)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(0))
    print("Training on: " + device.type)

    env = Connect4()
    agent1 = DQNAgent(device=device)
    try:
        agent1.load_model("model1.pth")
    except FileNotFoundError:
        print("No pre-trained model1 found")
    
    agent2 = DQNAgent(device=device)
    try:
        agent2.load_model("model2.pth")
    except FileNotFoundError:
        print("No pre-trained model2 found")

    #self_train(env, agent1, agent2, 5000)
    human_vs_ai(env, agent1, -1)
    #diverse_train(env, agent1, 50000, 20)

    agent1.save_model("model1.pth")
    agent2.save_model("model2.pth")
    print("Training complete")