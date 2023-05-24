# EMICoRe: Expected Maximum Improvement over Confident Regions
![Angler logo](doc/img/angler.png)

## Suggested Installation

Clone the repository (or extract from the supplement):
```bash
$ git clone <repository_URL> emicore
```
Move into the new repository and create a new environment:
```bash
$ cd emicore
$ python -m venv .venv
```

Install the emicore module with pip:
```bash
$ .venv/bin/pip install -e .
```

## Running the baselines (NFT)

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
      --cache cache.h5
```

## Running the standard LBFGS optimization:
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
      --cache cache.h5
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
      --cache cache.h5
```

Detailed hyperparameters for the experiments can be found in the appendix in section *F Experiment Details*
