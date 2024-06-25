![image](./logo_emicore.png)
# EMICoRe: Expected Maximum Improvement over Confident Regions

This is the repository for the NeurIPS 2023 paper ``Physics-Informed Bayesian Optimization of Variational Quantum 
Circuits`` [[1]](#1).

## Suggested Installation

Clone the repository (or extract from the supplement of our paper):
```bash
$ git clone git@github.com:angler-vqe/emicore.git emicore
```
Move into the repository's directory and create a new python environment:
```bash
$ cd emicore
$ python -m venv .venv
```

Install the emicore module with pip:
```bash
$ .venv/bin/pip install -e .
```
## Creating the training data

When comparing different optimization routines, it is important to make sure the difference in the performance is not 
due to some different initialization or seeds. In order to ensure every optimization starts from the same point and uses 
the same seed one can initialize different optimization schemes from the same initial points. To do so, the first step 
is to create a train set using the `generate.py` script
```bash
$ mkdir train_data
$ .venv/bin/python scripts/generate.py make-train \
      --seed 0 \
      train_data/train-ising-5-3-1024.h5 \
      --j-coupling -1,0,0 \
      --h-coupling 0,0,-1 \
      --n-qbits 5 \
      --n-layers 3 \
      --sector 1 \
      --circuit esu2 \
      --pbc False \
      --noise-level  0.0 \
      --n-readout 1024 \
      --train-samples 1 \
```

Once the training dataset has been generated it can be loaded for initializing any optimization (LBFGS, NFT and EMICoRe) 
using the `--train-data` option as shown in the examples below. One caveat is that the setup (couplings, qubits, 
layers, readout etc.) specified for creating the training set should coincide with the parameters of the optimization.

## Running the baselines (NFT)
The EMICoRe repository implements the state of the art Nakanishi-Fuji-Todo (NFT) algorithm [[2]](#2). This serves as the main 
baseline for the paper and can be run by calling the `nakanishi.py` script
```bash
$ .venv/bin/python scripts/nakanishi.py optimize \
      ising-5-3-1024-nftseq.h5 \
      --j-coupling -1,0,0 \
      --h-coupling 0,0,-1 \
      --n-qbits 5 \
      --n-layers 3 \
      --circuit esu2 \
      --pbc False \
      --n-readout 1024 \
      --n-iter 300 \
      --sequential \
      --train-data train_data/train-ising-5-3-1024.h5 
```

## Running the standard LBFGS optimization:
Similarly, one can run standard Bayesian Optimization using the LBFGS optimizer and any of the implemented kernels or
acquisition functions (EI, UCB) using the `bayesian_optimization.py` script
```bash
$ .venv/bin/python scripts/bayesian_optimization.py train \
      ising-5-3-1024-lbfgs.h5 \
      --j-coupling -1.0,0.0,0.0 \
      --h-coupling 0.0,0.0,-1.0 \
      --n-qbits 5 \
      --n-layers 3 \
      --circuit esu2 \
      --pbc no \
      --n-readout 1024 \
      --n-iter 300 \
      --kernel vqe \
      --acq-params 'func=ei,optim=lbfgs' \
      --hyperopt 'optim=grid,max_gamma=20,interval=100*1+20*9+10*100,steps=120,loss=mll' \
      --train-data train_data/train-ising-5-3-1024.h5 
```

## Running our approach (EMICoRe):

```bash
$ .venv/bin/python scripts/bayesian_optimization.py train \
      ising-5-3-1024-emicore.h5 \
      --j-coupling -1.0,0.0,0.0 \
      --h-coupling 0.0,0.0,-1.0 \
      --n-qbits 5 \
      --n-layers 3 \
      --circuit esu2 \
      --pbc no \
      --n-readout 1024 \
      --n-iter 300 \
      --kernel vqe \
      --acq-params 'func=ei,optim=emicore,pairsize=20,gridsize=100,corethresh=1.0,corethresh_width=10,samplesize=100,smo-steps=0,smo-axis=True' \
      --hyperopt 'optim=grid,max_gamma=20,interval=100*1+20*9+10*100,steps=120,loss=mll' \
      --train-data train_data/train-ising-5-3-1024.h5 
```
Detailed hyperparameters for reproducing the experiments of the paper can be found in the appendix in section 
**G Experiment Details**. In order to produce the longer runs from Fig. 8 in the Appendix of the NeurIPS paper, one can 
progressively discard older points in order to keep the points used to initialize the GP constant. To do this one can
use `--inducer` option
```bash
--inducer 'last_slack:retain=100:slack=20'
```
where `last_slack` indicates the criterion for choosing the inducing points while `retain` and `slack` are the number of 
points retained and discarded, respectively, when the number of measured points in the GP exceeds the sum of the two, 
e.g., when more than 120 observations are stored in the GP.

## BibTeX Citation

If you find this code related and/or useful for your own research please consider citing our NeurIPS paper:

```
@inproceedings{nicoli2023physics,
  author       = {Kim A. Nicoli and
                  Christopher J. Anders and
                  Lena Funcke and
                  Tobias Hartung and
                  Karl Jansen and
                  Stefan Kuhn and
                  Klaus{-}Robert M{\"{u}}ller and
                  Paolo Sornati and
                  Pan Kessel and
                  Shinichi Nakajima},
  title     = {Physics-Informed Bayesian Optimization of Variational Quantum Circuits},
  booktitle = {Advances in Neural Information Processing Systems 37: Annual Conference
               on Neural Information Processing Systems 2023, NeurIPS 2023, December
               10-16, 2023, New Orleans, LA, US},
  year      = {2023},
}

```

## References
<a id="1">[1]</a> 
K. A. Nicoli, C. J. Anders, et al. -
*Physics-Informed Bayesian Optimization of Variational Quantum Circuits*, 
Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2023, 
NeurIPS 2023, December 10-16, 2023, New Orleans, LA, US. (Dec 2023)

<a id="2">[2]</a> 
K. M. Nakanishi, K. Fujii, and S. Todo -
*Physics-Informed Bayesian Optimization of Variational Quantum Circuits*, 
Phys. Rev. Research 2, 043158 (Oct 2020)
