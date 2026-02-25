# DermDepth: Toward Monocular Metric Scale 3D Reconstruction Models for Dermatology

**MICCAI 2026**

Foundation depth models produce excellent relative geometry on dermatological images but systematically mis-estimate metric scale by 4-38x, making them unusable for clinical measurement. DermDepth corrects this by fine-tuning only the scale prediction head (2.1M of 331M parameters, <1%) of MoGe-2 on **D-Synth**, a new synthetic dermoscopic dataset with pixel-perfect 3D ground truth. Progressive refinement with small amounts of real clinical data (346 samples) achieves near-perfect metric scale across three diverse benchmarks, with equitable performance across Fitzpatrick skin tones.

## Key Results

| Method | SKINL2 Scale | WoundsDB Scale | DDI Ratio | Fairness Gap |
|--------|:---:|:---:|:---:|:---:|
| MoGe-2 (base) | 16.10x | 0.62x | 81.0x | 10.9x |
| DA3 | 4.16x | 0.67x | 53.6x | 34.9x |
| **DermDepth** | **0.87x** | **0.91x** | **1.95x** | **1.0x** |

Scale ratio target = 1.0x. Fairness gap target = 1.0x.

## Repository Structure

```
dermdepth/
├── code/
│   ├── analysis/             # Dataset exploration and baseline analysis
│   ├── annotation/           # DDI ruler annotation tools
│   ├── data_generation/      # D-Synth rendering and MoGe format conversion
│   │   ├── generate_dermdepth_dataset.py   # Main D-Synth generation script
│   │   ├── convert_to_moge.py              # S-SYNTH → MoGe format converter
│   │   ├── convert_eval_to_moge.py         # WoundsDB/SKINL2 → MoGe format
│   │   ├── create_ddi_training_data.py     # DDI pseudo-GT from ruler areas
│   │   └── depth_utils.py                  # Depth encoding and intrinsics utils
│   ├── evaluation/           # Metric evaluation scripts
│   │   ├── eval_depth.py                   # Main depth evaluation (scale, AbsRel, SI-d1)
│   │   ├── eval_ddi_rulers.py              # DDI ruler-based evaluation + fairness
│   │   ├── eval_normals.py                 # Surface normal evaluation
│   │   └── eval_baselines.py               # Baseline model evaluation
│   └── visualization/        # Paper figure generation
├── configs/                  # MoGe-2 training configs for all experiments
├── notebooks/
│   └── generate_dermdepth_colab.ipynb      # Colab notebook for D-Synth generation
├── scripts/                  # Evaluation shell scripts
└── LICENSE
```

## Setup

### Dependencies

DermDepth builds on [MoGe-2](https://github.com/microsoft/MoGe). Clone and install it first:

```bash
git clone https://github.com/microsoft/MoGe.git
cd MoGe && pip install -e .
```

Then install additional dependencies:

```bash
pip install accelerate mlflow-skinny
```

For D-Synth data generation, you also need [Mitsuba 3](https://mitsuba-renderer.org/) and [S-SYNTH assets](https://huggingface.co/datasets/didsr/ssynth_data).

### Data

Download evaluation datasets from their original sources:

- **SKINL2**: [Plenoptic dermoscopic depth](https://figshare.com/articles/dataset/SKINL2_-_Skin_Lesion_Light_Field_Dataset/6626502) (de Faria et al., 2019)
- **WoundsDB**: [Time-of-flight wound depth](http://www.ii.uj.edu.pl/~jusz/WoundsDB.html) (Juszczyk et al., 2020)
- **DDI**: [Diverse Dermatology Images](https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965) (Daneshjou et al., 2022)

Then prepare for training/evaluation:

```bash
# Convert evaluation datasets to MoGe format
python code/data_generation/convert_eval_to_moge.py --dataset skinl2 --input data/SKINL2 --output data/dermdepth_train/skinl2_moge
python code/data_generation/convert_eval_to_moge.py --dataset woundsdb --input data/DB_ALL --output data/dermdepth_train/woundsdb_moge

# Create DDI pseudo-GT from ruler annotations
python code/data_generation/create_ddi_training_data.py --input data/DDI --output data/dermdepth_train/ddi_moge
```

### Pretrained Model

Download the MoGe-2 pretrained weights from [HuggingFace](https://huggingface.co/Ruicheng/moge-2-vitl-normal):

```bash
cd MoGe && python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Ruicheng/moge-2-vitl-normal', 'pretrained_moge2.pt', local_dir='.')"
```

## Training

DermDepth uses progressive training with scale-head-only fine-tuning:

```bash
cd MoGe

# Stage 1: Synthetic-only (scale head, ~2.5h on 1x A100)
python -m moge.train --config ../configs/dermdepth_exp_a.json --workspace ../output/training/exp_a

# Stage 2: + Real data (WoundsDB + SKINL2)
python -m moge.train --config ../configs/dermdepth_exp_g.json --workspace ../output/training/exp_g

# Stage 3: + DDI pseudo-GT (best model)
python -m moge.train --config ../configs/dermdepth_exp_h.json --workspace ../output/training/exp_h
```

## Evaluation

```bash
# Evaluate on SKINL2
python code/evaluation/eval_depth.py --model MoGe/pretrained_moge2.pt --dataset skinl2 --split test

# Evaluate on WoundsDB
python code/evaluation/eval_depth.py --model MoGe/pretrained_moge2.pt --dataset woundsdb --split test

# Evaluate on DDI (with ruler GT and fairness analysis)
python code/evaluation/eval_ddi_rulers.py --model MoGe/pretrained_moge2.pt --split test

# Evaluate baselines (DA3, MapAnything, PPD)
python code/evaluation/eval_baselines.py --method da3 --dataset skinl2
```

## D-Synth Dataset

Generate synthetic dermoscopic training data with metric-scale 3D ground truth:

```bash
# Local generation (requires Mitsuba 3 + S-SYNTH assets)
python code/data_generation/generate_dermdepth_dataset.py --num_samples 3000 --output data/dermdepth_train/dsynth

# Or use the Colab notebook for cloud generation
# See notebooks/generate_dermdepth_colab.ipynb
```

## Citation

```bibtex
@inproceedings{dermdepth2025,
  title={DermDepth: Toward Monocular Metric Scale 3D Reconstruction Models for Dermatology},
  author={Anonymized},
  booktitle={International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2025}
}
```

## Acknowledgments

This work builds on:
- [MoGe-2](https://github.com/microsoft/MoGe) (Wang et al., 2025) for the base architecture
- [S-SYNTH](https://github.com/DIDSR/ssynth-release) (Kim et al., MICCAI 2024) for the synthetic skin rendering framework

## License

See [LICENSE](LICENSE) for details.
