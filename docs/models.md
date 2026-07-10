# Models

SuperHugeAAD implements two families of models for AAD: **deep neural networks** (DNNs) trained with gradient descent, and **linear models** solved analytically via sufficient statistics.

---

## DNN Models

All DNN models are `torch.nn.Module` subclasses wrapped by `MInterface`. They receive EEG input as `(batch, time, channels)` and output reconstructed envelopes or class logits.

### VLAAI

**File:** `superhuge/model/pure_cnn/vlaai.py`
**Config:** `models/vlaai.yaml`

A multi-block CNN architecture designed for EEG-to-speech-envelope regression:

```
Input: (B, T, C)
  │
  ├── Block 1
  │   ├── Extractor: Conv1d (constrained) → LayerNorm → ELU
  │   └── OutputContext: Conv1d → EinMix (linear projection)
  │
  ├── Block 2..N (same structure)
  │
  └── Output: (B, T', features)
```

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `nb_blocks` | 4 | Number of processing blocks |
| `extractor_args.num_kernels` | [256, 256, 256, 128, 128] | Kernel counts per layer |
| `extractor_args.kernel_sizes` | [8] | Temporal kernel sizes |
| `output_context_kernel_size` | 32 | Output context window size |
| `dropout` | 0.5 | Dropout rate |
| `use_skip` | false | Skip connections between blocks |

**Variants:**
- `VLAAI` — Standard version
- `VLAAI_ws` (`rebok_vlaai.yaml`) — Weight-sharing variant; same extractor parameters reused across blocks

### Deformer

**File:** `superhuge/model/former/deformer.py`
**Config:** `models/deformer.yaml`

A CNN + Transformer hybrid for EEG decoding:

```
Input: (B, T, C)
  │
  ├── Pre-Conv: Constrained Conv2D
  │
  ├── Transformer Encoder × MHA_DEPTH
  │   ├── Residual CNN Block
  │   ├── Multi-Head Self-Attention
  │   └── Feed-Forward Network
  │
  └── Output MLP: (B, T', features)
```

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_kernels` | 8 | CNN kernel count |
| `temporal_kernel_size` | 13 | Temporal kernel size |
| `mha_depth` | 2 | Number of transformer layers |
| `mha_embed_dim` | 32 | Attention embedding dimension |
| `mha_num_heads` | 2 | Number of attention heads |
| `ff_hidden_dim` | 64 | Feed-forward hidden dimension |
| `dropout` | 0.3 | Dropout rate |

**Variants:**
- `Deformer` — Standard version
- `Deformer_ws` (`rebok_deformer.yaml`) — Weight-sharing variant

### SSMamba

**File:** `superhuge/model/ssmamba/SSM2Mel.py`
**Config:** `models/ssmamba.yaml`

The most complex model, combining state-space models (S4), Bimamba, and attention:

```
Input: (B, T, C)
  │
  ├── Positional Encoding
  ├── Subject Embedding (one-hot → learnable projection)
  ├── Self-Attention
  ├── UNet with S4 Models
  │   ├── Downsampling path
  │   └── Upsampling path
  ├── Bimamba Blocks
  ├── External Attention
  └── Convolution Blocks with S4
```

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_inner` | 256 | Inner dimension |
| `n_head` | 2 | Attention heads |
| `n_layers` | 1 | Number of layers |
| `dropout` | 0.5 | Dropout rate |
| `within_sub_num` | 110 | Subject embedding dimension |

> **Platform note:** SSMamba uses `triton` and custom CUDA ops. It is **only importable on Linux/WSL**, not native Windows.

### SimpleCNN

**File:** `superhuge/model/pure_cnn/simple_cnn.py`
**Config:** `models/simple_cnn.yaml`

A minimal baseline for quick experiments:

```
ZeroPad2d → Conv2d → ReLU → Channel Reduction → Output
```

**Parameters:** `temporal_kernel_size=17`, `num_kernels=5`

### LSM-CNN

**File:** `superhuge/model/pure_cnn/lsm_cnn.py`
**Config:** `models/lsm_cnn.yaml`

Learnable Spatial Mapping CNN. Projects EEG channels to a 2D grid via `EinMix`, then applies 3D convolutions:

```
EEG (B, T, C)
  │
  ├── EinMix: C → (H × W)  (learnable 2D projection)
  ├── Conv3D on (B, T, H, W)
  └── Output
```

**Key parameters:** `lsm_chan_dim=8`, `cnn_num_layers=1`, `temporal_kernel_size=13`, `channel_kernel_size=3`, `num_kernels=5`

---

## Linear Models

Linear models are analytically solved (no gradient descent). They use sufficient statistics accumulated across the training set.

### Wiener Filter

**File:** `superhuge/model/linear/regression/wf.py`
**Config:** `models/wf.yaml`

The standard linear decoder for AAD. Constructs a lagged input matrix and solves the linear system:

```
R_xx @ W = R_xy
```

where `R_xx` is the auto-correlation of EEG and `R_xy` is the cross-correlation between EEG and speech envelope.

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `pre_lag` | 0.0 | Pre-stimulus lag in seconds |
| `post_lag` | 0.4 | Post-stimulus lag in seconds |
| `use_lwcov` | true | Use Ledoit-Wolf shrinkage for covariance |
| `l2` | 0.0 | L2 regularization strength |

**Sufficient statistics accumulation:** `WienerFilterState` accumulates `R_xx` and `R_xy` across batches in `stats` mode, then solves in `raw` mode.

### CCA (Canonical Correlation Analysis)

**File:** `superhuge/model/linear/regression/cca.py`

Finds linear projections of EEG and speech that maximize correlation.

### Filterbank CCA

**File:** `superhuge/model/linear/regression/fb_cca.py`

Applies CCA independently to multiple frequency bands and combines results.

### Riemannian Wiener Filter

**File:** `superhuge/model/linear/regression/rie_wf.py`

Applies Riemannian geometry to covariance matrices before Wiener filtering.

---

## Classification Models

### CSP Classifier

**File:** `superhuge/model/linear/classify/csp.py`

Common Spatial Patterns — finds spatial filters that maximize variance difference between classes.

### RGC Classifier

**File:** `superhuge/model/linear/classify/rgc.py`

Riemannian Geometry Classifier for spatial attention classification.

---

## Model Interfaces

Models are wrapped in Lightning interfaces that handle training, metrics, and logging:

| Interface | Purpose |
|-----------|---------|
| `MInterface` | Base — all models use this or a subclass |
| `RegressionInterface` | Envelope reconstruction — adds post_model, PCC/F1 metrics |
| `ClassifyInterface` | Classification — adds accuracy/AUC/macro-F1 metrics |
| `LinearRegressionInterface` | Combines `LinearInterface` + `RegressionInterface` |
| `ChannelMapping1DRegressionInterface` | 1D channel pre-mapping variant |
| `ChannelMapping2DRegressionInterface` | 2D channel pre-mapping variant |

## Model Input / Output Convention

All model `forward()` methods receive and return tensors with specific shapes:

**Regression models:**
```
Input:  eeg (B, T, C), [env (B, T, F, 1), meta]
Output: eeg_hat (B, T', F), [env (B, T', F, 1), ...]
```
Where `F` is the number of audio features (1 for envelope).

**Classification models:**
```
Input:  eeg (B, T, C), [meta]
Output: logits (B, num_classes)
```

`MInterface` auto-detects input/output specifications via `inspect.signature()` and return type annotations.
