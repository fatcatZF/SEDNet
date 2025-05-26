import torch 
import torch.nn as nn 

import numpy as np 

import gymnasium as gym 

import joblib
import itertools

import fnmatch

from drift_detectors import MLPDriftDetector

import argparse

import os 


class EnsembleMLPDriftDetector(nn.Module):
        def __init__(self, models, discrete_action):
            super(EnsembleMLPDriftDetector, self).__init__()
            self.models = models 
            self.discrete_action = discrete_action

        def forward(self, x):
            with torch.no_grad():
                outputs = []
                for model in self.models:
                    model["model"].eval()

                    if self.discrete_action:
                        transition = x[:,:-1].detach().numpy()
                        action = x[:, -1:]
                        transition_scaled = torch.from_numpy(model["scaler"].transform(transition)).float()
                        x = torch.cat([transition_scaled, action], dim=-1)
                    else:
                        x = model["scaler"].transform(x.numpy()) 
                        x = torch.from_numpy(x).float()
                    outputs.append(torch.sigmoid(model["model"](x)))
                return torch.mean(torch.stack(outputs), dim=0).detach().item()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="cartpole", help="name of environment")
    parser.add_argument("--policy-type", type=str, default="dqn", help="type of rl policy")
    args = parser.parse_args()

    allowed_envs = {"cartpole", "lunarlander", "hopper", 
                    "halfcheetah", "humanoid"}
    
    allowed_policy_types = {"dqn", "ppo", "sac"}

    env_dict = {
      "cartpole" : "CartPole-v1",
      "lunarlander": "LunarLander-v3",
      "hopper": "Hopper-v5",
      "halfcheetah": "HalfCheetah-v5",
      "humanoid": "Humanoid-v5"
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

    policy_env_name = args.policy_type + '-' + args.env
    model_folder = os.path.join("./trained_models", policy_env_name)
    os.makedirs(model_folder, exist_ok=True)

    # List and filter matching subdirectories
    matching_dirs = [       
        name for name in os.listdir(model_folder)
        if os.path.isdir(os.path.join(model_folder, name)) and fnmatch.fnmatch(name, "single_[0-9]")
    ]

    number_of_trained = len(matching_dirs)

    assert number_of_trained >= 6 

    env = gym.make(env_dict[args.env])

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




    combinations = list(itertools.combinations(range(6), 5))

    for idx, combo in enumerate(combinations):
        models = []
        print(f"Creating ensemble_{idx} with models: {combo}")

        for i in combo:
            model_path = os.path.join(model_folder, f"single_{i}", "model.pth")
            scaler_path = os.path.join(model_folder, f"single_{i}", "scaler.joblib")

            model = torch.load(model_path, weights_only=False)
            scaler = joblib.load(scaler_path)
            #mu = scaler.mean_
            #std = np.sqrt(scaler.var_)
            #print(f"mu: {mu}, std:{std}")
            models.append({"model": model, "scaler":scaler})
        
        ensemble = EnsembleMLPDriftDetector(models, discrete_action)

        ensemble_path = os.path.join(model_folder, f"ensemble_{idx}.pth")
        torch.save(ensemble, ensemble_path)







if __name__ == "__main__":
    main()










    


