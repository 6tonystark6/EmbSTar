# [Efficient Multi-branch Black-box Semantic-aware Targeted Attack Against Deep Hashing Retrieval](https://ieeexplore.ieee.org/abstract/document/10889223) (ICASSP 2025)
By Chihan Huang, Xiaobo Shen

Deep hashing have achieved exceptional performance in retrieval tasks due to their robust representational capabilities. However, they inherit the vulnerability of deep neural networks to adversarial attacks. These models are susceptible to finely crafted adversarial perturbations that can lead them to return incorrect retrieval results. Although numerous adversarial attack methods have been proposed, there has been a scarcity of research focusing on targeted black-box attacks against deep hashing models. We introduce the Efficient Multi-branch Black-box Semantic-aware Targeted Attack against Deep Hashing Retrieval (EmbSTar), capable of executing targeted black-box attacks on hashing models. Initially, we distill the target model to create a knockoff model. Subsequently, we devised novel Target Fusion and Target Adaptation modules to integrate and enhance the semantic information of the target label and image. Knockoff model is then utilized to align the adversarial image more closely with the target image semantically. With the knockoff model, we can obtain powerful targeted attacks with few queries. Extensive experiments demonstrate that EmbSTar significantly surpasses previous models in its targeted attack capabilities, achieving SOTA performance for targeted black-box attacks.

## Code Organization
```
project-root/
├── attacked_methods/     # attacked model folder
│   ├── CSQ               # CSQ
│   │   └── CSQ.py        # CSQ
│   ├── DPSH              # DPSH
│   │   └── DPSH.py       # DPSH
│   └── HashNet           # HashNet
│       └── HashNet.py    # HashNet
├── attack_model.py       # EmbSTar class main code
├── attacked_model.py     # load attacked model
├── hashing_train.py      # train the attacked hashing model
├── load_data.py          # dataset configuration
├── main.py               # code for training EmbSTar
├── model.py              # components of EmbSTar
├── README.md             # readme
└── utils.py              # utils
```

# Requirements

- python == 3.8.20
- pytorch == 2.0.0
- torchvision == 0.15.1
- numpy == 1.24.4
- h5py == 3.11.0
- scipy == 1.10.1
- tqdm == 4.67.1
- pandas == 2.0.3


## Use

### Train deep hashing models
Initialize the hyper-parameters in hashing_train.py, and then run
```Python
python hashing_train.py
```
and the checkpoints will bbe saved in '/attacked_models/DPSH_FLICKR_32/DPSH.pth', where DPSH and FLICKR are the attacked model and dataset you choose.

### Train knockoff model

```Python
python main.py --knockoff True --kb 32 --ke 100 --kbz 24
```

### Train EmbSTar

```Python
python main.py --dataset FLICKR --attacked_method DPSH --train --test --bit 32 --batch_size 24 --n_epochs 50 --n_epochs_decay 100 --learning_rate 1e-4 --output_dir DPSH_FLICKR_32
```


## Citation
```
@INPROCEEDINGS{10889223,
  author={Huang, Chihan and Shen, Xiaobo},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Efficient Multi-branch Black-box Semantic-aware Targeted Attack Against Deep Hashing Retrieval}, 
  year={2025},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10889223}}
```
