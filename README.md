This repository shows an example of utilizing CO2 within [Fairseq](https://github.com/facebookresearch/fairseq).

--------------------------------------------------------------------------------

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq-CO2** and develop locally:

``` bash
git clone https://github.com/weigao266/fairseq-CO2.git
cd fairseq-CO2
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with `--ipc=host` or `--shm-size`
 as command line options to `nvidia-docker run` .

# Usage

Run the script `run_co2_local.sh` to train a GPT-2 (Medium) model with 355M parameters, using CO2. The script sets `co2_base_algorithm = localsgd`, and `co2_outer_momentum = 0.2`.

``` bash
cd co2_examples
bash run_co2_local.sh
```


# Citation

If you find our work useful, please cite the following paper:

``` bibtex
@article{sun2024co2,
  title={CO2: Efficient Distributed Training with Full Communication-Computation Overlap},
  author={Sun, Weigao and Qin, Zhen and Sun, Weixuan and Li, Shidi and Li, Dong and Shen, Xuyang and Qiao, Yu and Zhong, Yiran},
  journal={arXiv preprint arXiv:2401.16265},
  year={2024}
}
```
