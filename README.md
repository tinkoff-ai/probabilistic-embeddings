# Probabilistic Embeddings
This repository contains an implementation of all
Probabilistic Metric Learning (PML) approaches
from [Probabilistic Embeddings Revisited](https://arxiv.org/pdf/2202.06768.pdf) paper.
It fully supports the following probabilistic methods from previous works:
* [HIB](https://arxiv.org/abs/1810.00319)
* [PFE](https://arxiv.org/abs/1904.09658)
* [DUL-reg](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chang_Data_Uncertainty_Learning_in_Face_Recognition_CVPR_2020_paper.pdf)
* [DUL-cls](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chang_Data_Uncertainty_Learning_in_Face_Recognition_CVPR_2020_paper.pdf)
* [SCF](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Spherical_Confidence_Learning_for_Face_Recognition_CVPR_2021_paper.pdf)
* [vMF-FL](https://arxiv.org/abs/1706.04264)
* [vMF-loss](https://openaccess.thecvf.com/content/ICCV2021/papers/Scott_von_Mises-Fisher_Loss_An_Exploration_of_Embedding_Geometries_for_Supervised_ICCV_2021_paper.pdf)
* [SPE](https://arxiv.org/abs/1909.11702)
* [DSML](https://arxiv.org/abs/1802.09662)

In addition to PML approaches, classical (deterministic) Metric Learning (ML)
methods are supported:
* [ArcFace](https://arxiv.org/abs/1801.07698)
* [CosFace](https://arxiv.org/abs/1801.09414)
* [Proxy-Anchor Loss](https://arxiv.org/abs/2003.13911)
* [Multi-Similarity Loss](https://arxiv.org/abs/1904.06627)
* [Proxy-NCA](https://arxiv.org/pdf/1703.07464.pdf)

## Getting Started

### Installation

1. Clone this repository:
    ```bash
    git clone git@github.com:tinkoff-ai/probabilistic-embeddings.git
    cd probabilistic-embeddings
    ```
2. We recommend building our Docker image with `Dockerfile`.
3. Library must be installed before execution. It is recommended to use editable installation:
    ```bash
    pip install -e .
    ```
4. You can check the installation using tests:
    ```bash
    tox -e py38 -r
    ```

### Quick Start

1. Prepare experiment `.yaml` config. In this example, a simple
ArcFace model is trained on LWF dataset:

   ```yaml
   dataset_params:
     name: lfw-openset
     samples_per_class: null
   
   model_params:
     # Embedder maps input image to embedding vector space.
     embedder_params:
       model_type: resnet18
       # Use ImageNet pretrain.
       pretrained: true
     distribution_params:
       # Spherical 512D embeddings.
       spherical: true
       dim: 512
     # For deterministic embeddings specify Dirac distribution (default).
     distribution_type: dirac
     classifier_type: arcface
   
   trainer_params:
     optimizer_type: adam
     optimizer_params:
       lr: 3.0e-4
   ```
2. Download and unpack [LFW](http://vis-www.cs.umass.edu/lfw/) dataset.
3. Run training with command:
   ```bash
   python3 -m probabilistic_embeddings train \
   --config <path-to-yaml-config> \
   --train-root <logs-and-checkpoints-root> \
   <path-to-lfw-data-root>
   ```
4. Logs and checkpoints will be saved to `./<logs-and-checkpoints-root>`.
The default logging format is Tensorboard.

## WanDB support

To enable WanDB logging run the experiment with command:
```bash
WANDB_ENTITY=<entity-name> \
WANDB_API_KEY=<api-key> \
CUDA_VISIBLE_DEVICES=<gpu-index> \
python3 -m probabilistic_embeddings train \
--config <path-to-yaml-config> \
--logger wandb:<project-name>:<experiment-name> \
--train-root <logs-and-checkpoints-root> \
<path-to-dataset-root>
```

## Supported commands

### Training

`train` runs standard training pipeline:
```bash
python3 -m probabilistic_embeddings train \
--config <path-to-yaml-config> \
--train-root <logs-and-checkpoints-root> \
<path-to-data-root>
```

To apply K-fold cross-validation scheme use `cval` command.

### Evaluation

`test` computes metrics for a given checkpoint:
```bash
CUDA_VISIBLE_DEVICES=<gpu-index> \
python3 -m probabilistic_embeddings test \
--config <path-to-config> \
--checkpoint <path-to-checkpoint> \
<path-to-data-root>
```

`evaluate` performs model evaluation over multiple random seeds.
Add `num_evaluation_seeds` field to experiment config to specify number of random seeds.
Use `evaluate-cval` command to evaluate with cross-validation. Add `num_validation_folds`
to `dataset_params` to set the number of folds.

### Hyperparameter tuning

In order to run WanDB sweeps, use `hopt`and `hopt-cval` commands.
Hyperparameter tuning is only supported with WanDB logger.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> \
python3 -m probabilistic_embeddings hopt \
--config <path-to-config> \
--logger wandb:<project-name>:<experiment-name> \
--train-root <training-root> <path-to-data-root>
```

Hyperparameters to search and their ranges should be specified in
config as in this example:

```yaml
...
model_params:
  ...
  classifier_type: arcface
  classifier_params:
    _hopt:
      scale:
        min: 1.0
        max: 64.0
      margin:
        min: 0.0
        max: 1.0
...
```

## Reproducing paper results

In order to reproduce all the results of the paper, you need to generate configs for all experiments:
```bash
mkdir configs/reality/generated
python scripts/configs/generate-reality.py \
configs/reality/templates/ \
configs/reality/generated/ \
--best configs/reality/best/
```
Our hyperparameter search results are stored in `configs/reality/best`.
You can reproduce hyperparameter search with `hopt` command and download best parameters from WanDB.
To reproduce training and evaluation, please, refer to the commands above.

## Supported Datasets

Repository supports multiple datasets.
Face recognition:
[MS1MV2](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_),
[MS1MV3](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_),
[LFW](http://vis-www.cs.umass.edu/lfw/), and
[CASIA](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_).
Image retrieval:
[Cars196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html),
[CUB200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html),
[In-shop clothes (Inshop)](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)
and [Stanford Online Products (SOP)](https://cvgl.stanford.edu/projects/lifted_struct/).
We also implement multiple image classification datasets, please,
refer to `./src/probabilistic_embeddings/dataset` for more details.

## Citing
If you use code from this repository in your project, please, cite our paper:
```bibtex
@inproceedings{pml2022,
  title={Probabilistic Embeddings Revisited},
  author={Ivan Karpukhin and Stanislav Dereka and Sergey Kolesnikov},
  year={2022},
  url={https://arxiv.org/pdf/2202.06768.pdf}
```
