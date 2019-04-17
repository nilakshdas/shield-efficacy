# Reproducing Differentiable SLQ Attack on SHIELD [[paper](https://arxiv.org/abs/1902.00541)]

### SETUP
To get this repository with the attached submodules, run the following command:
```
git clone --recursive https://github.com/nilakshdas/shield-extended.git
```

### TOY DATASET AND MODELS
Get the toy dataset and the model checkpoints from [this link](https://gatech.box.com/s/hdzmw8lv4c0jcqud8xxesa3uaexo3fq6) and place them in the `scratch` directory.

### USAGE
```
python main.py --attack_models 80,60 --eval_models 80,60
```
