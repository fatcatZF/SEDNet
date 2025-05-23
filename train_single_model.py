import os 

import fnmatch

import numpy as np 

import torch
import torch.nn as nn 
import torch.optim as optim 

import gymnasium as gym 

from stable_baselines3 import PPO, SAC, DQN 

from datetime import datetime

import joblib

from data_utils import generate_data
from drift_detectors import MLPDriftDetector


import argparse






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="cartpole", help="name of environment")
    parser.add_argument("--policy-type", type=str, default="dqn", help="type of rl policy")
    parser.add_argument("--env-steps", type=int, default=20000, help="Steps of running training environment")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    
    args = parser.parse_args()


    allowed_envs = {"cartpole", "lunarlander", "hopper", 
                    "halfcheetah", "humanoid"}
    
    allowed_policy_types = {"dqn", "ppo", "sac"}

    env_dict = {
      "cartpole" : "CartPole-v1",
      "lunarlander": "LunarLander-v3",
      "hopper": "Hopper-v5",
      "halfcheetah": "HalfCheetah-v3",
      "humanoid": "Humanoid-v5"
    }

    env_agent_dict = {
      "cartpole" : "dqn-cartpole",
      "lunarlander": "ppo-lunarlander",
      "hopper": "ppo-hopper",
      "halfcheetah": "sac-halfcheetah",
      "humanoid": "sac-humanoid"
    }
    
    
    env_action_discrete = {
        "cartpole": True,
        "lunarlander": True,
        "hopper": False,
        "halfcheetah": False,
        "humanoid": False
    }



    
    if args.env not in allowed_envs:
        raise NotImplementedError(f"The environment {args.env} is not supported.")
    if args.policy_type not in allowed_policy_types:
        raise NotImplementedError(f"The policy {args.policy_type} is not supported.")

    print("Parsed arguments: ")
    print(args) 

    if (args.policy_type=="dqn"):
        AGENT = DQN 
    elif (args.policy_type=="ppo"):
        AGENT = PPO 
    else:
        AGENT = SAC 

    policy_env_name = args.policy_type + '-' + args.env

    ### Load Trained agent
    agent_path = os.path.join('./agents/', policy_env_name)
    agent = AGENT.load(agent_path) 
    print("Successfully Load Trained Agent.")


    model_folder = os.path.join("./trained_models", policy_env_name)

    os.makedirs(model_folder, exist_ok=True)

    pattern = "single*"

    # List and filter matching subdirectories
    matching_dirs = [
        name for name in os.listdir(model_folder)
        if os.path.isdir(os.path.join(model_folder, name)) and fnmatch.fnmatch(name, pattern)
    ]

    number_of_trained = len(matching_dirs)

    save_folder_name = os.path.join(model_folder, f"single{number_of_trained}")
    os.makedirs(save_folder_name)

    scaler_path = os.path.join(save_folder_name, "scaler.joblib")
    model_path = os.path.join(save_folder_name, "model.pth")

    env = gym.make(env_dict[args.env])

    scaler, train_loader, test_loader = generate_data(agent, env)

    joblib.dump(scaler, scaler_path) # save the scaler 

    discrete_action = env_action_discrete[args.env]
    if discrete_action:
        num_actions = env.action_space.n
        action_embed_dim = int(min(50, (num_actions**0.25)*4))
    else:
        num_actions = env.action_space.sample().shape[-1]
        action_embed_dim = None

    model = MLPDriftDetector(
        obs_dim = env.observation_space.sample().shape[-1],
        num_actions = num_actions,
        discrete_action = discrete_action,
        hidden_dim = 256,
        action_embed_dim = action_embed_dim
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9,
                                                     patience=10, verbose=False)
    
    num_epochs = args.epochs

    for epoch in range(num_epochs):
        train_losses = []
        best_test_loss = float('inf')
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
           
            train_losses.append(loss.item())

        test_losses = []
        with torch.no_grad():
            model.eval()
            for batch_idx, (data, target) in enumerate(test_loader):
                outputs = model(data)
                loss = criterion(outputs, target)
                test_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)
        scheduler.step(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model, model_path)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()

    