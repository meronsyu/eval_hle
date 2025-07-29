# 松尾研LLM開発コンペ2025 開発コード・手順の標準化

評価コードです。現在、HLEタスクのカテゴリを選択して評価できるように修正しています。
  
**「サーバ利用手順」は、サーバ利用における重要な注意事項も含まれるため必ずご確認ください。**  
**「LLM評価手順」は、開発したLLMのランキングに使うため必ずご確認ください。**

## サーバ利用手順

サーバにログインしてジョブシステムを利用する手順です。  
[こちら](https://docs.google.com/document/d/16KKkFM8Sbqx0wgcCY4kBKR6Kik01T-jn892e_y67vbM/edit?tab=t.0)

## 評価の実行方法
### 実行前・二つのタブを開いてください
sshでログインノード→scancelで自動投入ジョブキャンセル、からのsrun
```python
bash ../shareP12/scancel_hatakeyama.sh gpu84 gpu85 && srun --job-name=evaluate_phi4 --partition P12 --nodes=1 --nodelist osk-gpu[84] --gpus-per-node=1 --time=12:00:00 --pty bash -i
```
```python
conda activate llmbench
```
### vllm起動側
eval_hle/conf/config.yaml

```python
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
#CUDA_VISIBLE_DEVICESでトレーニングに使用するGPUの数を制御します。
#例えば、単一GPUの場合は以下のように設定します：
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
#※AMD製のGPUではないため、ROCR_VISIBLE_DEVICES を指定しないようにしてください。指定するとエラーになります。
unset ROCR_VISIBLE_DEVICES

ulimit -v unlimited
ulimit -m unlimited
```
```python
model: microsoft/Phi-4-reasoning-plus
```
モデル名を評価したいモデルに変更

### 評価実行側

```python
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
#CUDA_VISIBLE_DEVICESでトレーニングに使用するGPUの数を制御します。
#例えば、単一GPUの場合は以下のように設定します：
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
#※AMD製のGPUではないため、ROCR_VISIBLE_DEVICES を指定しないようにしてください。指定するとエラーになります。
unset ROCR_VISIBLE_DEVICES

ulimit -v unlimited
ulimit -m unlimited
```
```python
category_filter:
  - "Biology/Medicine"
```
```python
category_filter: null
```
カテゴリを複数選ぶかnullを記述

## 注意点

ベースコードのhle_script.shをうまく実行できなかったので、vllmの起動とタスクの回答、評価を分けています。

## Contributors

```
```

