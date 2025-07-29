#!/bin/bash

# --- Slurm ジョブ設定 ---
#SBATCH --job-name=nyancat
#SBATCH --partition=P12
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 # GPUが必要な場合
#SBATCH --time=14:00:00 # 実行に時間がかかる可能性を考慮して設定
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# --- 実行コマンド ---
python predict.py
