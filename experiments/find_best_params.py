#!/usr/bin/env python3
import os
import csv
import json
from pathlib import Path

def find_best_parameters():
    """
    Scans the ray_results directory for all training trials, reads their progress,
    and returns the hyperparameters of the trial that achieved the highest reward.
    """
    # Get the absolute path to the ray_results directory
    script_dir = Path(__file__).parent.absolute()
    ray_results = script_dir / 'ray_results'
    
    best_reward = -float('inf')
    best_params = None
    best_trial = None

    if not ray_results.exists():
        print(f'Directory not found: {ray_results}')
        return

    print(f"Scanning {ray_results} for best PPO parameters...\n")

    # Iterate over all PPO_* trial directories
    for trial_dir in ray_results.rglob('PPO_*'):
        if trial_dir.is_dir() and (trial_dir / 'progress.csv').exists():
            try:
                with open(trial_dir / 'progress.csv', 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    
                    if header and 'env_runners/episode_reward_mean' in header:
                        idx = header.index('env_runners/episode_reward_mean')
                        
                        for row in reader:
                            if len(row) > idx and row[idx]:
                                try:
                                    rew = float(row[idx])
                                    if rew > best_reward:
                                        best_reward = rew
                                        best_trial = trial_dir.name
                                        
                                        # Read the parameters for this best trial
                                        config_path = trial_dir / 'params.json'
                                        if config_path.exists():
                                            with open(config_path, 'r') as cf:
                                                best_params = json.load(cf)
                                except ValueError:
                                    pass # Ignore rows with non-float rewards
            except Exception as e:
                print(f"Warning: Could not read {trial_dir.name}: {e}")

    # Output the results
    if best_params:
        print("🏆 BEST RESULT FOUND 🏆")
        print("="*40)
        print(f"Trial Folder: {best_trial}")
        print(f"Peak Reward:  {best_reward:.2f}")
        print("-" * 40)
        print("Best Hyperparameters:")
        
        # Print standard hyperparameters
        keys_to_print = ['lr', 'gamma', 'entropy_coeff', 'train_batch_size', 'sgd_minibatch_size']
        for k in keys_to_print:
            print(f"  {k}: {best_params.get(k)}")
            
        # Print environment specific parameters if available
        env_config = best_params.get('env_config', {})
        if env_config:
            print(f"  num_episodes: {env_config.get('num_episodes')}")
            
        print("="*40)
    else:
        print('No valid training results found in the progress.csv files.')

if __name__ == "__main__":
    find_best_parameters()
