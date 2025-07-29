#!/bin/bash
#SBATCH --job-name=Phi4-reasoning-plus
#SBATCH --partition=P012
#SBATCH --nodelist=osk-gpu[84]
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --time=14:00:00
#SBATCH --output=/home/Competition2025/adm/X006/logs/%x-%j.out
#SBATCH --error=/home/Competition2025/adm/X006/logs/%x-%j.err
#SBATCH --export=OPENAI_API_KEY=<openaiのkeyを入力>
#--- モジュール & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0  
module load nccl/2.24.3 
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench

# Hugging Face 認証
export HF_TOKEN=<ここにトークンを入れて>
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"                   # デバッグ用

#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i 0,1,2,3,4 -l 3 > nvidia-smi.log &
pid_nvsmi=$!

#--- vLLM 起動（2GPU）----------------------------------------------
# tensor-parallel-sizeについてはmulti headsを割り切れる数に指定する必要あり
# どこでモデルのmulti headsを見れるかの手法はこちら
# 
vllm serve microsoft/Phi-4-reasoning-plus \
  --tensor-parallel-size 2 \
  --reasoning-parser deepseek_r1 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --max-model-len 131072 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.80 
  > vllm.log 2>&1 &
pid_vllm=$!

#--- ヘルスチェック -------------------------------------------------
until curl -s http://127.0.0.1:8000/health >/dev/null; do
  echo "$(date +%T) vLLM starting …"
  sleep 10
done
echo "vLLM READY"