import numpy as np
from collections import deque

import gymnasium as gym 


from sklearn.metrics import roc_auc_score

from stable_baselines3 import PPO, SAC, DQN 

import torch 
import joblib

import os 
import glob 
import pickle
import json 
from datetime import datetime 

from environment_util import make_env 

from drift_detectors import MLPDriftDetector



import argparse 

from river import drift 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="cartpole", help="name of environment")
    parser.add_argument("--policy-type", type=str, default="dqn", help="type of rl policy")
    parser.add_argument("--model-type", type=str, default="single", help="type of drift detector models")
    parser.add_argument("--env0-steps", type=int, default=1000, help="Validation Steps")
    parser.add_argument("--env1-steps", type=int, default=3000, help="Undrifted Steps")
    parser.add_argument("--env2-steps", type=int, default=3000, help="Semantic Drift Steps")
    parser.add_argument("--env3-steps", type=int, default=3000, help="Noisy Drift Steps")
    parser.add_argument("--n-exp-per-model", type=int, default=10, 
                        help="number of experiments of each trained model.") 

    args = parser.parse_args() 

    allowed_envs = {"cartpole", "mountaincar", "lunarlander", "hopper", 
                    "halfcheetah", "humanoid"}
    
    allowed_policy_types = {"dqn", "ppo", "sac"}
    
    if args.env not in allowed_envs:
        raise NotImplementedError(f"The environment {args.env} is not supported.")
    if args.policy_type not in allowed_policy_types:
        raise NotImplementedError(f"The policy {args.policy_type} is not supported.")

    print("Parsed arguments: ")
    print(args) 

    env_dict = {
      "cartpole" : "CartPole-v1",
      "mountaincar": "MountainCar-v0",
      "lunarlander": "LunarLander-v3",
      "hopper": "Hopper-v5",
      "halfcheetah": "HalfCheetah-v5",
      "humanoid": "Humanoid-v5"
    }

    # Load trained Agent
    if (args.policy_type=="dqn"):
        AGENT = DQN 
    elif (args.policy_type=="ppo"):
        AGENT = PPO 
    else:
        AGENT = SAC 

    policy_env_name = args.policy_type + '-' + args.env

    agent_path = os.path.join('./agents/', policy_env_name)
    agent = AGENT.load(agent_path) 
    print("Successfully Load Trained Agent.")

    env = gym.make(env_dict[args.env])


    env_action_discrete = {
        "cartpole": True,
        "mountaincar":True, 
        "lunarlander": True,
        "hopper": False,
        "halfcheetah": False,
        "humanoid": False
    }


    model_folder = os.path.join("trained_models", args.policy_type+'-'+args.env)
    if args.model_type == "single":
        pattern = "single_[0-9]"
    elif args.model_type == "single_noise":
        pattern = "single_noise_[0-9]"
    elif args.model_type == "single_drift":
        pattern = "single_drift_[0-9]"
    else:
        pattern = "ensemble_[0-9]"

    

    discrete_action = env_action_discrete[args.env]
    if discrete_action:
        num_actions = env.action_space.n
        action_embed_dim = int(min(50, (num_actions**0.25)*4))
    else:
        num_actions = env.action_space.sample().shape[-1]
        action_embed_dim = None
    
    # Load Drift Detector Models
    loaded_models = []
    model_pattern = os.path.join(model_folder, pattern)
    print(model_pattern)
    matching_models = glob.glob(model_pattern)

    print(matching_models) 
    if len(matching_models)==0:
        raise NotImplementedError(f"There is no trained model for the environment {args.env}.")
    

    for model_path in matching_models:
        model = MLPDriftDetector(
            obs_dim = env.observation_space.sample().shape[-1],
            num_actions = num_actions,
            discrete_action = discrete_action,
            hidden_dim = 256,
            action_embed_dim = action_embed_dim
        )
        model = torch.load(os.path.join(model_path, "model.pth"), 
                           weights_only=False)
        scaler = joblib.load(os.path.join(model_path, "scaler.joblib"))

        loaded_models.append({"model":model, "scaler":scaler})

    print(f"Number of trained models: {len(loaded_models)}")

    ## Create environments
    env0, env1, env2, env3 = make_env(name=args.env)
    print("Successfully create environments")


    result = dict()
    #for i in range(len(loaded_models)):
    #    result[f"{args.model_type}_{i}"] = dict()      
    ## Run the evaluations
    auc_semantic_values = []
    auc_noise_values = []

    ph_delays_sem = []
    ph_fas_sem = []
    ph_delays_noise = []
    ph_fas_noise = []

    ad_delays_sem = []
    ad_fas_sem = []
    ad_delays_noise = []
    ad_fas_noise = []

    ks_delays_sem = []
    ks_fas_sem = []
    ks_delays_noise = []
    ks_fas_noise = []
 
    for i, model in enumerate(loaded_models):
            for j in range(args.n_exp_per_model):

                # Validation
                scores_val = []
                env_current = env0
                obs_t, _ = env_current.reset()
                for t in range(args.env0_steps):
                    action_t, _states = agent.predict(obs_t, deterministic=True)
                    obs_tplus1, r_tplus1, terminated, truncated, info = env_current.step(action_t)
                    done = terminated or truncated
                    transition = np.concatenate([obs_t, obs_tplus1-obs_t], axis=-1)
                    transition = transition.reshape(1, -1)
                    if discrete_action:
                        transition = model["scaler"].transform(transition)
                        x = np.concatenate([transition, action_t.reshape(1,-1)], axis=-1)
                        x_tensor = torch.from_numpy(x).float()
                    else:
                        x = np.concatenate([transition, action_t.reshape(1,-1)], axis=-1)
                        x = model["scaler"].transform(x)
                        x_tensor = torch.from_numpy(x).float()
                    
                    with torch.no_grad():
                        model["model"].eval()
                        y_predict = torch.sigmoid(model["model"](x_tensor))[0].detach().item()
                    scores_val.append(y_predict)

                    obs_t = obs_tplus1
                    if done:
                        obs_t, _ = env_current.reset()


                mu_val = np.mean(scores_val)
                std_val = np.std(scores_val)

                
                # Semantic Drift
                scores_sem = []
                env_current = env1 
                obs_t, _ = env_current.reset()
                total_steps = args.env1_steps + args.env2_steps
                for t in range(1, total_steps+1):
                    action_t, _state = agent.predict(obs_t, deterministic=True)
                    obs_tplus1, r_tplus1, terminated, truncated, info = env_current.step(action_t)
                    done = terminated or truncated
                    transition = np.concatenate([obs_t, obs_tplus1-obs_t], axis=-1)
                    transition = transition.reshape(1, -1)
                    if discrete_action:
                        transition = model["scaler"].transform(transition)
                        x = np.concatenate([transition, action_t.reshape(1,-1)], axis=-1)
                        x_tensor = torch.from_numpy(x).float()
                    else:
                        x = np.concatenate([transition, action_t.reshape(1,-1)], axis=-1)
                        x = model["scaler"].transform(x)
                        x_tensor = torch.from_numpy(x).float()
                    
                    with torch.no_grad():
                        model["model"].eval()
                        y_predict = torch.sigmoid(model["model"](x_tensor))[0].detach().item()
                    scores_sem.append((y_predict-mu_val)/(std_val+1e-6))

                    obs_t = obs_tplus1
                    if done:
                        obs_t, _ = env_current.reset()
                    if t==args.env1_steps:
                        env_current = env2 
                        obs_t, _ = env_current.reset()
                
                
                y_env1 = np.zeros(args.env1_steps)
                y_env2 = np.ones(args.env2_steps)
                y = np.concatenate([y_env1, y_env2])
                auc_semantic = roc_auc_score(y, scores_sem)
                auc_semantic_values.append(auc_semantic)


                # Page-Hinkley Semantic
                ph = drift.PageHinkley(mode='up', delta=0.005)
                fa = 0
                delay = args.env2_steps + 1000
                for t, val in enumerate(scores_sem):
                    ph.update(val)
                    if ph.drift_detected and val > 0:
                        if t < args.env1_steps:
                            fa += 1
                        if t >= args.env1_steps:
                            delay = t - args.env1_steps
                            break 

                ph_delays_sem.append(delay)
                ph_fas_sem.append(fa)

                # ADWIN Semantic
                adwin = drift.ADWIN()
                fa = 0
                delay = args.env2_steps + 1000
                for t, val in enumerate(scores_sem):
                    adwin.update(val)
                    if adwin.drift_detected and val>0:
                        if t<args.env1_steps:
                           fa+=1
                        if t>=args.env1_steps:
                           delay = t-args.env1_steps
                           break
                
                ad_delays_sem.append(delay)
                ad_fas_sem.append(fa)


                # KSWIN semantic
                kswin = drift.KSWIN(window=scores_val)
                fa = 0
                delay = args.env2_steps+1000
                for t, val in enumerate(scores_sem):
                    kswin.update(val)
                    if kswin.drift_detected and val>0:
                        if t < args.env1_steps:
                            fa += 1
                        if t >= args.env1_steps:
                            delay = t - args.env1_steps
                
                ks_delays_sem.append(delay)
                ks_fas_sem.append(fa)





                # Noise Drift
                scores_noise = []
                env_current = env1 
                obs_t, _ = env_current.reset()
                total_steps = args.env1_steps + args.env3_steps
                for t in range(1, total_steps+1):
                    action_t, _state = agent.predict(obs_t, deterministic=True)
                    obs_tplus1, r_tplus1, terminated, truncated, info = env_current.step(action_t)
                    done = terminated or truncated
                    transition = np.concatenate([obs_t, obs_tplus1-obs_t], axis=-1)
                    transition = transition.reshape(1,-1)
                    if discrete_action:
                        transition = model["scaler"].transform(transition)
                        x = np.concatenate([transition, action_t.reshape(1,-1)], axis=-1)
                        x_tensor = torch.from_numpy(x).float()
                    else:
                        x = np.concatenate([transition, action_t.reshape(1,-1)], axis=-1)
                        x = model["scaler"].transform(x)
                        x_tensor = torch.from_numpy(x).float()
                    
                    with torch.no_grad():
                        model["model"].eval()
                        y_predict = torch.sigmoid(model["model"](x_tensor))[0].detach().item()
                    scores_noise.append((y_predict-mu_val)/(std_val+1e-6))

                    obs_t = obs_tplus1
                    if done:
                        obs_t, _ = env_current.reset()
                    if t==args.env1_steps:
                        env_current = env3
                        obs_t, _ = env_current.reset()


                y_env1 = np.zeros(args.env1_steps)
                y_env3 = np.ones(args.env3_steps)
                y = np.concatenate([y_env1, y_env3])
                auc_noise = roc_auc_score(y, scores_noise)
                auc_noise_values.append(auc_noise)


                # Page Hinkley Noise
                ph = drift.PageHinkley(mode="up", delta=0.005)
                fa = 0
                delay = args.env3_steps+1000
                for t, val in enumerate(scores_noise):
                    ph.update(val)
                    if ph.drift_detected and val>0:
                        if t < args.env1_steps:
                            fa += 1
                        if t >= args.env1_steps:
                            delay = t - args.env1_steps
                            break 

                ph_delays_noise.append(delay)
                ph_fas_noise.append(fa)


                # ADWIN Noise
                adwin = drift.ADWIN()
                fa = 0
                delay = args.env3_steps+1000
                for t, val in enumerate(scores_noise):
                    adwin.update(val)
                    if adwin.drift_detected and val>0:
                        if t<args.env1_steps:
                           fa+=1
                        if t>=args.env1_steps:
                           delay = t-args.env1_steps
                           break
            
                ad_delays_noise.append(delay)
                ad_fas_noise.append(fa)


                ## Noise
                kswin = drift.KSWIN(window=scores_val)
                fa = 0
                delay = args.env3_steps+1000
                for t, val in enumerate(scores_noise):
                    kswin.update(val)
                    if kswin.drift_detected and val>0:
                        if t < args.env1_steps:
                            fa += 1
                        if t >= args.env1_steps:
                            delay = t - args.env1_steps

                ks_delays_noise.append(delay)
                ks_fas_noise.append(fa)
                





    result["auc_semantic_mean"] = np.mean(auc_semantic_values)
    result["auc_noise_mean"] = np.mean(auc_noise_values)

    result["ph_delays_semantic_mean"] = np.mean(ph_delays_sem)
    result["ph_delays_noise_mean"] = np.mean(ph_delays_noise)
    result["ph_fas_semantic_mean"] = np.mean(ph_fas_sem)
    result["ph_fas_noise_mean"] = np.mean(ph_fas_noise)

    result["ad_delays_semantic_mean"] = np.mean(ad_delays_sem)
    result["ad_delays_noise_mean"] = np.mean(ad_delays_noise)
    result["ad_fas_semantic_mean"] = np.mean(ad_fas_sem)
    result["ad_fas_noise_mean"] = np.mean(ad_fas_noise)

    result["ks_delays_semantic_mean"] = np.mean(ks_delays_sem)
    result["ks_delays_noise_mean"] = np.mean(ks_delays_noise)
    result["ks_fas_semantic_mean"] = np.mean(ks_fas_sem)
    result["ks_fas_noise_mean"] = np.mean(ks_fas_noise)

    


    result_folder = os.path.join("./results", args.env)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder) 
    print("results folder", result_folder)
    result_file = f"{args.model_type}-{args.env}.json"
    print("result file: ", result_file)

    result_path = os.path.join(result_folder, result_file)
    print("result path: ", result_path) 

    with open(result_path, 'w') as f:
        json.dump(result, f, separators=(',', ':'))


if __name__ == "__main__":
    main()