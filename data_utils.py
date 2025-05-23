import numpy as np

import torch 
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler

import gymnasium as gym
from gymnasium.spaces import Discrete, Box

from stable_baselines3.common.base_class import BaseAlgorithm


def generate_data(agent: BaseAlgorithm, env: gym.Env,
                  total_steps: int = 20000):
    actions = []
    actions_drift = []
    transitions = []
    transitions_drift = []
    labels = []
    labels_drift = []

    obs_t, _ = env.reset()

    for _ in range(total_steps):
        action_t, _states = agent.predict(obs_t, deterministic=True)
        actions.append(action_t)
        actions_drift.append(action_t)
        obs_tplus1, r_tplus1, terminated, truncated, info = env.step(action_t)

        done = terminated or truncated

        # undrifted transiton
        transitions.append(np.concatenate([obs_t, obs_tplus1-obs_t], axis=-1))
        labels.append(0)

        # synthetic drifted example
        obs_prime = env.observation_space.sample()
        drift_rate = np.random.uniform(low=0.0, high=0.05)
        obs_tplus1_prime = (1-drift_rate)*obs_tplus1 + drift_rate*obs_prime + np.random.normal(scale=0.005,
                                                                                           size=obs_t.shape)
        
        transitions_drift.append(np.concatenate([obs_t, obs_tplus1_prime-obs_t]))
        labels_drift.append(1)

        obs_t = obs_tplus1
        if done:
            obs_t, _ = env.reset()

    
    transitions = np.array(transitions)
    actions = np.array(actions)
    transitions_drift = np.array(transitions_drift)
    actions_drift = np.array(actions_drift)

    transitions_all = np.concatenate([transitions, transitions_drift], axis=0)
    actions_all = np.concatenate([actions, actions_drift], axis=0)
    labels_all = np.concatenate([labels, labels_drift], axis=0)

    # split the data
    idx = np.random.permutation(len(transitions_all))
    transitions_all = transitions_all[idx]
    actions_all = actions_all[idx]
    labels_all = labels_all[idx]

    n_train = int(len(transitions_all)*0.7)
    transitions_train = transitions_all[:n_train]
    actions_train = actions_all[:n_train]
    y_train = labels_all[:n_train]

    transitions_test = transitions_all[n_train:]
    actions_test = actions_all[n_train:]
    y_test = labels_all[n_train:]


    scaler = StandardScaler()

    if isinstance(env.action_space, Discrete):
        transitions_train_scaled = scaler.fit_transform(transitions_train)
        transitions_test_scaled = scaler.transform(transitions_test)
        X_train = np.concatenate([transitions_train_scaled, 
                                  actions_train.reshape(-1,1)], axis=-1)
        X_test = np.concatenate([transitions_test_scaled, 
                                 actions_test.reshape(-1,1)], axis=-1)

    else:
        X_train = np.concatenate([transitions_train, actions_train], axis=-1)
        X_test = np.concatenate([transitions_test, actions_test], axis=-1)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.tensor(y_train).float().unsqueeze(-1)
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.tensor(y_test).float().unsqueeze(-1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    return scaler, train_loader, test_loader





