# HDTR-Net-v2.0

TODO:
-  Demo videos
-  pre-trained model
-  code for testing
-  code for training
-  code for preprocess dataset
-  guideline 
-  arxiv paper release


HDTR-Net-v2.0 Will Come Soon!

UPDATE:
- 2025-01-03: Train VQVAE , Teeth Enhance pipeline!



## Training
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python runer/train.py --cfg ./opt/vqgan.yml --mode vqgan


CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python runer/train.py --cfg ./opt/teeth_enhance.yml --mode teeth_enhance

