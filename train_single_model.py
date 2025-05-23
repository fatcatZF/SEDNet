import os 

import fnmatch

import numpy as np 

import gymnasium as gym 

from stable_baselines3 import PPO, SAC, DQN 

from datetime import datetime

import joblib

from data_utils import generate_data


import argparse




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
    model_path = os.path.join(save_folder_name, "model.pt")

    env = gym.make(env_dict[args.env])

    scaler, train_loader, test_loader = generate_data(agent, env)





if __name__ == "__main__":
    main()

    