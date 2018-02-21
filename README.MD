# Memorability transfer
Code for reproducting result from a paper https://arxiv.org/abs/1704.01745.
## Lamem (Scorer)
1) Train a scorer. Scorer is based on caffe. Training procedures is identical to one in memnet. Just split training part of the dataset (first split) (http://memorability.csail.mit.edu/download.html) in 2 parts to train 2 independent scorers. You can find proto files in lamem folder. You can download caffe model from https://yadi.sk/d/059Y2cii3SQ39L. Put trained models to lamem folder.

## Texture nets (Synthesizer)
2) Train sythethiser. Synthethiser is based on torch. 

2.0) Clone https://github.com/DmitryUlyanov/texture_nets. 

2.1) Merge folder from this repository with their folder. 

2.2) Create folder texture_nets/datasets/abstract_art_seeds put top50 and bottom50 memorable images from devian-art(https://www.dropbox.com/sh/3obloy1yp21izw3/AAC18wpp0Iuz6RHLzkfxEbt6a?dl=0) into this folder. This images will be used as styles for style transfer.

2.3) Create folder texture_nets/datasets/abstract_art_test. Put 10k images from lamem test split to this folder. This images will be used for training and testing Seed selector.

2.4) Create folder texture_nets/datasets/lamem/train and texture_nets/datasets/lamem/val as described in https://github.com/DmitryUlyanov/texture_nets. This images is used by texture network framework and can be arbitrary.

2.5) run ```./train_models.sh abstract_art 2``` for alpha equal to 2. This can take uo to one week, to train on style networks on a single gpu. This will train all models, create all images, and score all images with internal predictor. 

2.6) modify ```eval.lua``` to use external predictor and compute score for all the images aggain. (in train_models.sh abstract_art comment ```th train.lua``` and ```th test_dir.lua```

2.7) Memorability scores after this stage can be downloaded from https://yadi.sk/d/059Y2cii3SQ39L

## Seed Selector
3) Train a seed selector. Seed selector is based on theano/lasagne.

3.1) Convert HybrydCNN to lasagne. Or download it from https://yadi.sk/d/059Y2cii3SQ39L alex_start_net.npy. You can also use vgg16 from same folder.

3.2) Train model with ```python train.py```. See list of parameters in this file.

3.3) Test with ```python eval_external.py```. See list of parameters in that file.


