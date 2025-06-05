# [U-Mamba](https://wanglab.ai/u-mamba.html)

**U-Mamba: 生体医用画像セグメンテーションにおける長距離依存性の強化**の公式リポジトリです。  
最新情報を受け取りたい方は [メーリングリスト](https://forms.gle/bLxGb5SEpdLCUChQ7) にご登録ください。


## 🔧 インストール手順

**動作環境要件**: `Ubuntu 20.04`, `CUDA 11.8`

1. 仮想環境を作成：
   ```bash
   conda create -n umamba python=3.10 -y
   conda activate umamba
   ```

2. [PyTorch 2.0.1](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) をインストール：
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
   ```

3. [Mamba](https://github.com/state-spaces/mamba) 関連ライブラリをインストール：
   ```bash
   pip install causal-conv1d>=1.2.0
   pip install mamba-ssm --no-cache-dir
   ```

4. コードを取得：
   ```bash
   git clone https://github.com/bowang-lab/U-Mamba
   ```

5. モジュールをインストール：
   ```bash
   cd U-Mamba/umamba
   pip install -e .
   ```

---

## Docker+condaの環境構築

### Condaの有効化
```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate umamba
```

### Mamba関連モジュールのインストール
```bash
cd /U-Mamba/umamba
pip install -e .
```

### conda仮想環境から抜ける
```bash
conda deactivate
```


問題
umambaのversionがtorch2.1とかを要求してきた
⇒元々はtorchのバージョンは2.0.6とかだった

🔸 問題①: pip install mamba-ssm が失敗
原因: PyTorch 2.7 (CUDA 12.6) がbuildに使われ、DockerベースCUDA 11.8と不一致でビルド失敗。

対応:

PyTorchを 2.1.2+cu118 に固定（CUDA 11.8 対応の最終バージョン）

mamba-ssm はこのバージョンでビルド成功

transformerのpipで入れることでまあましに？とりあえずimport mamba-ssmは通るように



やること
pip install -e . の自動化
⇒これができないとコンテナ壊れた時に忘れる

🔸 問題②: pip install -e . が失敗
原因: Dockerfile内で umamba ディレクトリに移動して pip install -e . を実行するが、該当ディレクトリがDockerコンテナ内に存在しない

対応:

ホスト側の /U-Mamba をコンテナにマウントしてから pip install -e .

または、git clone して WORKDIR を設定してから同コマンド実行


---

## ✅ 動作確認（sanity test）

Pythonのインタラクティブ環境で以下を実行：

```python
import torch
import causal_conv1d 
import mamba_ssm
print("CUDA available:", torch.cuda.is_available())
print("CUDA version from torch:", torch.version.cuda)
print("CUDA device name:", torch.cuda.get_device_name(0))
print("causal_conv1d and mamba_ssm loaded successfully")
```


![network](https://github.com/bowang-lab/U-Mamba/blob/main/assets/U-Mamba-network.png)



https://github.com/bowang-lab/U-Mamba/assets/19947331/1ac552d6-4ffd-4909-ba31-7b48644fd104


---


## 🚀 モデルの学習

データセットは[こちら](https://drive.google.com/drive/folders/1DmyIye4Gc9wwaA7MVKFVi-bWD2qQb-qN?usp=sharing)からダウンロードし、`data` フォルダに配置してください。  
U-Mambaは [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) フレームワーク上に構築されています。  
独自のデータセットを使用する場合は、この[ガイドライン](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md)に従って準備してください。


### 🔄 前処理

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### 📘 2Dモデルの学習

- `U-Mamba_Bot` の学習：

```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaBot
```

- `U-Mamba_Enc` の学習：

```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaEnc
```

### 📕 3Dモデルの学習

- `U-Mamba_Bot` の学習：

```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerUMambaBot
```

- `U-Mamba_Enc` の学習：

```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerUMambaEnc
```


## 🔍 推論（inference）

- `U-Mamba_Bot` を使った予測：

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -f all -tr nnUNetTrainerUMambaBot --disable_tta
```

- `U-Mamba_Enc` を使った予測：

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -f all -tr nnUNetTrainerUMambaEnc --disable_tta
```

※ `CONFIGURATION` には `2d` または `3d_fullres` を指定してください。



## 💬 補足

1. **パスの設定**  
U-Mambaのデフォルトのデータディレクトリは `U-Mamba/data` です。  
既にnnUNetを使用しているユーザーは、以下のように `umamba/nnunetv2/path.py` を編集することで、別のディレクトリ構成に対応できます：

```python
base = '/home/ユーザー名/Documents/U-Mamba/data'
nnUNet_raw = join(base, 'nnUNet_raw')
nnUNet_preprocessed = join(base, 'nnUNet_preprocessed')
nnUNet_results = join(base, 'nnUNet_results')
```

2. **AMP（自動混合精度）によるNaNの問題**  
AMPが有効な場合、MambaモジュールでNaNが発生する可能性があります。  
AMP無効バージョンのトレーナーも提供されています：[こちら](https://github.com/bowang-lab/U-Mamba/blob/main/umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainerUMambaEncNoAMP.py)

## 📄 論文引用情報

```
@article{U-Mamba,
    title={U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation},
    author={Ma, Jun and Li, Feifei and Wang, Bo},
    journal={arXiv preprint arXiv:2401.04722},
    year={2024}
}
```

## 🙏 謝辞

本研究で使用されている公開データセットの全ての著者に感謝いたします。  
また、[nnU-Net](https://github.com/MIC-DKFZ/nnUNet) および [Mamba](https://github.com/state-spaces/mamba) の開発者にも、貴重なコードの公開に感謝します。

