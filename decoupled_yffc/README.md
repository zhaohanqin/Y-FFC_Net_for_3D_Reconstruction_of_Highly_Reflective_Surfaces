# Decoupled Y-FFC Copy

This directory contains a decoupled copy generated from the repository-root `Y_FFC_unified.py` and `train.py`.

The original root files were not modified.

## Layout

- `Y_FFC_unified.py`: compatibility entrypoint for the split copy.
- `train.py`: training entrypoint copy with the original configuration block.
- `yffc/models/common.py`: small shared model helpers.
- `yffc/models/ffc/`: Fast Fourier Convolution modules.
- `yffc/models/blocks/`: reusable convolution, multiscale, and upsampling blocks.
- `yffc/models/attention/`: attention gates, cross-modal fusion, and specular-aware attention.
- `yffc/models/branches/`: auxiliary model branches such as the ViT encoder branch.
- `yffc/models/heads/`: high-frequency enhancement and output refinement heads.
- `yffc/models/architecture/`: Encoder, decoders, enhancement layers, and `YFFC`.
- `yffc/models/network.py`: compatibility exports for code that imports from the old model file.
- `yffc/losses/`: metrics, basic losses, specular losses, and composite losses.
- `yffc/losses.py`: compatibility exports for old loss imports.
- `yffc/data/`: datasets and paired augmentations.
- `yffc/data.py`: compatibility exports for old data imports.
- `yffc/training/`: training utilities, optimization, metrics, loops, visualization, reporting, and enhanced training orchestration.
- `yffc/cli.py`: CLI compatible with the original unified script.

## Usage

Run from this directory if you want to use the decoupled copy directly:

```powershell
cd decoupled_yffc
python train.py
```

```powershell
cd decoupled_yffc
python Y_FFC_unified.py test
```

Or import it as a normal package from the repository root:

```powershell
python -c "import decoupled_yffc; from decoupled_yffc.yffc.models import YFFC; print('ok')"
```

Compatibility imports are still supported when `decoupled_yffc` is on `PYTHONPATH`:

```powershell
python -c "import sys; sys.path.insert(0, 'decoupled_yffc'); from yffc.models.network import YFFC; print('ok')"
```
