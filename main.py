import torch

from game.connect4_env import Connect4Env
from engine.model import DQNAgent
from engine.model_trainer import Trainer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Connect4Env()
    agent = DQNAgent(device=device)
    try:
        agent.load_model("model1.pth")
    except FileNotFoundError:
        print("No pre-trained model1 found")
    trainer = Trainer(env, agent, device)

    trainer.self_play(20, 100000)
    #trainer.human_vs_ai(1)
