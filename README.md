# Ego-VPA: Egocentric Video Understanding with Parameter-efficient Adaptation
[Ego-VPA: Egocentric Video Understanding with Parameter-efficient Adaptation](https://arxiv.org/pdf/2407.19520)  
[Tz-Ying Wu](http://www.svcl.ucsd.edu/people/gina), [Kyle Min](https://sites.google.com/view/kylemin), [Subarna Tripathi](https://subarnatripathi.github.io/), [Nuno Vasconcelos](http://www.svcl.ucsd.edu/~nuno)  
IEEE Winter Conference on Applications of Computer Vision (WACV), 2025

Ego-VPA is a lightweight adaption method for video foundation models. It employs a local sparse approximation of features to synthesize prompts with a cross-frame and cross-modal prompt basis, which models context fusion and cross-modal transfer efficiently. While using less learnable parameters (only 0.84%), Ego-VPA substantially improves over baselines and reaches the performance of full fine-tuning.


<p align="center">
  <img src="https://github.com/user-attachments/assets/c26d6528-4f2f-4381-ad29-a8083e51fc40" height=300px>
</p>

## Installation
The codebase is built atop [LaViLa](https://github.com/facebookresearch/LaViLa) and the dependencies remain the same. We provide `requirements.txt` for a easy setup.
```
$ pip install -r requirements.txt
```

## Data Preparation
Please follow the [instructions](https://github.com/facebookresearch/LaViLa/tree/main/datasets) to download CharadesEgo, EGTEA, and EPIC-Kitchens-100 datasets.

## Training
We provide config files for different datasets and model variations, e.g., `configs/charades_ego/prompt/ego_vpa.yml` defines the configurations for training Ego-VPA with the Charades-Ego dataset.
```
$ torchrun --rdzv_backend=c10d --rdzv_endpoint=<rdzv_endpoint> --nnodes=1 --nproc_per_node=<n_gpu> main_finetune_retrieval.py --config <config> --wandb
```

## Evaluation
The config files for evaluating each dataset can be found in `configs/<dataset>/test.yml`.
```
$ python main_finetune_retrieval.py --config <config> --evaluate
```

## Cite
If you find this repository useful, please consider citing our paper.
```
@inproceedings{Wu25Ego-VPA,
 title = {Ego-VPA: Egocentric Video Understanding with Parameter-efficient Adaptation},
 author = {Wu, Tz-Ying and Min, Kyle and Tripathi, Subarna and Vasconcelos, Nuno},
 booktitle = {IEEE Winter Conference on Applications of Computer Vision},
 year = {2025}
}
```
