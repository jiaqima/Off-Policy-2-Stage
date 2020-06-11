# Off-Policy-2-Stage

This repo provides a PyTorch implementation of the MovieLens experiments for the following paper:

[Off-policy Learning in Two-stage Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/3366423.3380130)

Jiaqi Ma, Zhe Zhao, Xinyang Yi, Ji Yang, Minmin Chen, Jiaxi Tang, Lichan Hong, Ed H. Chi. TheWebConf (WWW) 2020.

## Requirements
See `environment.yml`. Run `conda op2s_env create -f environment.yml` to install the required packages.

## Run the code
Example: `python run.py --loss_type loss_2s`.

The "Cross-Entropy", "1-IPS", and "2-IPS" objectives respectively correspond to "loss_ce", "loss_ips", and "loss_2s" in the code.

The MovieLens-1M dataset can be found on the [GroupLens website](https://grouplens.org/datasets/movielens/1m/).

## Cite

```
@inproceedings{ma2020off,
  title={Off-policy Learning in Two-stage Recommender Systems},
  author={Ma, Jiaqi and Zhao, Zhe and Yi, Xinyang and Yang, Ji and Chen, Minmin and Tang, Jiaxi and Hong, Lichan and Chi, Ed H},
  booktitle={Proceedings of The Web Conference 2020},
  pages={463--473},
  year={2020}
}
```
