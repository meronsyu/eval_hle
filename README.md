# 松尾研LLM開発コンペ2025 開発コード・手順の標準化

評価コードです。現在、HLEタスクのカテゴリを選択して評価できるように修正しました。
時間内に評価が終わらなくても、回答が作成されるように修正中です。

## 注意
環境構築前提です。
eval_hle/hle_script.shのOpenai_keyとhuggingfaceのトークンを自分で埋めてください。


## 評価の実行方法
### 実行前
GPU数を設定
```python
bash ../shareP12/scancel_hatakeyama.sh gpu84 gpu85 && srun --job-name=evaluate_phi4 --partition P12 --nodes=1 --nodelist osk-gpu[84] --gpus-per-node=1 --time=12:00:00 --pty bash -i
```
```python
conda activate llmbench
```

モデル名を評価したいモデルに変更
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
eval_hle/confのconfig.yamlを自分が評価したいモデルに変更
```python
model: microsoft/Phi-4-reasoning-plus
```

eval_hle/confのconfig.yamlを自分が評価したいカテゴリに変更
```python
category_filter:
  - "Biology/Medicine"
```
```python
category_filter: null
```
カテゴリを複数選ぶかnullを記述

### 実行時
モデルが立ち上がるまで待機。nohup.outの中身を見ながら待つ。
```python
nohup ./hle_script.sh > vllm.log 2>&1 &
```
立ち上がったモデルを元に評価
```python
nohup ./hle_prediction.sh > prediction.log 2>&1 &
```

## 注意点
慣れるまではエラーが出ると思うので、shellスクリプトを実行するときは&を消して、ログを確認するのをお勧めします。
gpu数は、モデルごとにmulti head数と語彙数でassertエラーが起きる可能性があるので、基本２の倍数がお勧めです。
https://www.notion.so/239e14b94af5807f88d0df0189e3cc98 にmulti head数の見方が載っています。

chmod +x hle_prediction.sh
chmod +x hle_script.sh
を忘れずに
## Contributors

```
```

