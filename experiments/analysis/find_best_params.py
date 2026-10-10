#!/usr/bin/env python3
import os
import csv
import json
from pathlib import Path

def find_best_parameters():
    """
    Scans persistent results history and ray_results directory for all training trials,
    and returns the hyperparameters of the trial that achieved the highest reward.
    """
    script_dir = Path(__file__).parent.absolute()
    exp_dir = script_dir.parent
    summary_json = exp_dir / 'logs' / 'experiments_summary.json'
    ray_results = exp_dir / 'ray_results'
    
    best_reward = -float('inf')
    best_params = None
    best_trial = None

    # 1. Try reading persistent experiment summary
    if summary_json.exists():
        try:
            with open(summary_json, 'r', encoding='utf-8') as f:
                entries = json.load(f)
            for e in entries:
                rw = e.get('eval_reward_mean')
                if not isinstance(rw, (int, float)):
                    rw = e.get('train_reward_mean')
                if isinstance(rw, (int, float)) and float(rw) > best_reward:
                    best_reward = float(rw)
                    best_trial = e.get('experiment_id')
                    best_params = {
                        'lr': e.get('learning_rate'),
                        'gamma': e.get('gamma'),
                        'entropy_coeff': e.get('entropy_coeff'),
                        'train_batch_size': e.get('batch_size'),
                        'env_config': {'num_episodes': e.get('num_episodes')}
                    }
        except Exception:
            pass

    if ray_results.exists():
        for trial_dir in ray_results.rglob('PPO_*'):
            if trial_dir.is_dir() and (trial_dir / 'progress.csv').exists():

                seen_trials.add(trial_dir.resolve())

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
