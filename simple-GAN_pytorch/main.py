"""
코드 참고        (ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/1.%20SimpleGAN/fc_gan.py
참고 영상        (ref) https://youtu.be/OljTVUVzPpM
코딩 스타일 참고 (ref) https://github.com/DoranLyong/U-NET-tutorial/blob/main/U-NET_pytorch/train.py
"""

#%% 

from tqdm import tqdm 
import hydra   # for handling yaml (ref) https://neptune.ai/blog/how-to-track-hyperparameters
from omegaconf import DictConfig
import torch 
import torch.nn as nn  # 학습 가능한 레이어들을 담은 패키지 ; # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.backends.cudnn as cudnn    # https://hoya012.github.io/blog/reproducible_pytorch/
                                        # https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
import torch.optim as optim # 최적화 알고리즘을 담은 패키지 ; # For all Optimization algorithms, SGD, Adam, etc.
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard


from models import ( Generator, 
                    Discriminator
                    )
from utils import get_loaders                    

"""

랜덤 발생 기준을 시드로 고정함. 

그러면 shuffle=True 이어도, 언제나 동일한 방식으로 섞여서 동일한 데이터셋을 얻을 수 있음. 

(ref) https://github.com/DoranLyong/VGG-tutorial/blob/main/VGG_pytorch/VGG_for_CIFAR10.py
"""

SEED = 42 # set seed 

torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED) # for multi-gpu



#%%

def train_fn(train_loader, model, optimizer, loss_fn, scaler, DEVICE, BATCH_SIZE, NUM_EPOCHS, cur_epoch):
    pass 





#%% 

@hydra.main(config_name='./cfg.yaml')

def main(cfg: DictConfig):


    """ Set your device 
    """

    gpu_no = 0  # gpu_number 

    DEVICE = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() and cfg.default.DEVICE.CUDA else 'cpu')

    print(f"device: { DEVICE }")



    """ Initialize the network 
    """    

    disc = Discriminator(cfg.hyperparams.IMAGE_DIM).to(DEVICE)

    gen = Generator(cfg.hyperparams.Z_DIM, cfg.hyperparams.IMAGE_DIM).to(DEVICE)

    print(f"{disc}")

    print(f"{gen}")



    """ Get dataloader 
    """    

    latent_noise  = torch.randn((cfg.hyperparams.BATCH_SIZE, cfg.hyperparams.Z_DIM)).to(DEVICE)  # fixed noise ; shape := [batch_size, z_dim]

    transforms = A.Compose(
        [
            A.Normalize(
                mean=[0.5, ],
                std=[0.5, ],
            ),
            ToTensorV2(), # Albumentations to torch.Tensor
        ],
    )


    dataset = datasets.MNIST(root="./dataset/", transform=transforms, download=True)                                    
    


#%% 

if __name__ == '__main__':
    main() 