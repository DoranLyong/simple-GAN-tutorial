"""
코드 참고        (ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/1.%20SimpleGAN/fc_gan.py
참고 영상        (ref) https://youtu.be/OljTVUVzPpM
코딩 스타일 참고 (ref) https://github.com/DoranLyong/U-NET-tutorial/blob/main/U-NET_pytorch/train.py
"""
#%% 

import torch 
import torch.nn as nn  # 학습 가능한 레이어들을 담은 패키지 ; # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.backends.cudnn as cudnn    # https://hoya012.github.io/blog/reproducible_pytorch/
                                        # https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
import torch.optim as optim # 최적화 알고리즘을 담은 패키지 ; # For all Optimization algorithms, SGD, Adam, etc.

from utils import ( Generator, 
                    Discriminator
                    )


# %%
