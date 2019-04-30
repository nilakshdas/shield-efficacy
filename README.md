# Reproducing Differentiable SLQ Attack on SHIELD [[paper](https://arxiv.org/abs/1902.00541)]

### SETUP
To get this repository with the attached submodules, run the following command:
```
git clone --recursive https://github.com/nilakshdas/shield-extended.git
```

Then, install the requirements:
```
pip install -r requirements.txt
pip install -r jobby/requirements.txt
```

### TOY DATASET AND MODELS
Get the toy dataset and the model checkpoints from [this link](https://gatech.box.com/s/hdzmw8lv4c0jcqud8xxesa3uaexo3fq6) and place them in the `scratch` directory.

### USAGE
```
python main.py --attack_models old-60,old-40,old-20 --eval_models old-80,old-60,old-40,old-20
```
