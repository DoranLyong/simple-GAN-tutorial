"""
코드 참고        (ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/1.%20SimpleGAN/fc_gan.py
참고 영상        (ref) https://youtu.be/OljTVUVzPpM
코딩 스타일 참고 (ref) https://github.com/DoranLyong/U-NET-tutorial/blob/main/U-NET_pytorch/train.py
"""

#%% 
import sys
import os
import os.path as osp

from tqdm import tqdm 
from colorama import Back, Style # assign color options on your text(ref) https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
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
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard


from models import ( Generator, 
                    Discriminator
                    )
from utils import get_loaders                    



""" Path checking 
"""
python_ver = sys.version
script_path = os.path.abspath(__file__)
cwd = os.getcwd()
os.chdir(cwd) #changing working directory 

print(f"Python version: {Back.GREEN}{python_ver}{Style.RESET_ALL}")
print(f"The path of the running script: {Back.MAGENTA}{script_path}{Style.RESET_ALL}")
print(f"CWD is changed to: {Back.RED}{cwd}{Style.RESET_ALL}")



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

def train_fn(loader, disc, gen, opt_disc, opt_gen, loss_fn, scaler, DEVICE, BATCH_SIZE, NUM_EPOCHS, cur_epoch):

    # (ref) https://github.com/DoranLyong/DeepLearning-model-factory/blob/master/ML_tutorial/PyTorch/Basics/lr_scheduler_tutorial.py
    loop = tqdm(enumerate(loader), total=len(loader)) 

    for batch_idx, (real, _) in loop: # 미니배치 별로 iteration 

        real = real.view(-1, 784).to(DEVICE)  # MNIST image; 28x28 = 784


    




#%% 

@hydra.main(config_name='./cfg.yaml')

def main(cfg: DictConfig):

    """ Set your device 
    """
    gpu_no = 0  # gpu_number 
    DEVICE = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() and cfg.default.DEVICE.CUDA else 'cpu')
    print(f"device: { DEVICE }")


    """ Get dataloader 
    """    
    latent_noise  = torch.randn((cfg.hyperparams.BATCH_SIZE, cfg.hyperparams.Z_DIM)).to(DEVICE)  # fixed noise ; shape := [batch_size, z_dim]

    transform  = transforms.Compose(
        [
            transforms.ToTensor(), # 순서가 중요함 (이게 먼저 와야함); (ref) https://discuss.pytorch.org/t/typeerror-img-should-be-pil-image-got-class-torch-tensor/85834
            transforms.Normalize( mean=[0.5, ], std=[0.5, ], ),
            
        ],
        )


    dataset = datasets.MNIST(root=osp.join(cwd, 'dataset'), transform=transform, download=True)    # HTTP Error 503: (ref) https://stackoverflow.com/questions/66646604/http-error-503-service-unavailable-when-trying-to-download-mnist-data
                                                                                        # (ref) https://bbdata.tistory.com/8
                                                                                        # (ref) https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.MNIST
    loader = DataLoader(dataset, batch_size= cfg.hyperparams.BATCH_SIZE, shuffle=True)


    """ Initialize the network 
    """    
    disc = Discriminator(cfg.hyperparams.IMAGE_DIM).to(DEVICE)
    gen = Generator(cfg.hyperparams.Z_DIM, cfg.hyperparams.IMAGE_DIM).to(DEVICE)
    print(f"{disc}")
    print(f"{gen}")


    disc = torch.nn.DataParallel(disc)  # 데이터 병렬처리          # (ref) https://tutorials.pytorch.kr/beginner/blitz/data_parallel_tutorial.html
                                        # 속도가 더 빨라지진 않음   # (ref) https://tutorials.pytorch.kr/beginner/former_torchies/parallelism_tutorial.html
                                        # 오히려 느려질 수 있음    # (ref) https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b
    gen = torch.nn.DataParallel(gen)


    cudnn.benchmark = True    



    """ Gradient Scaling
        (ref) https://pytorch.org/docs/stable/amp.html#gradient-scaling
    """        
    scaler = torch.cuda.amp.GradScaler()      



    """ Loss and optimizer  
    """
    opt_disc = optim.Adam(disc.parameters(), lr=cfg.hyperparams.LEARNING_RATE )
    opt_gen = optim.Adam(gen.parameters(), lr=cfg.hyperparams.LEARNING_RATE)

    loss_fn = nn.BCELoss()



    """ Tensorboard 
    """
    writer_fake = SummaryWriter(f"logs/fake")
    writer_real = SummaryWriter(f"logs/real")
    step = 0    



    """ Start the training-loop
    """
    for epoch in range(cfg.hyperparams.NUM_EPOCHS):

        # Run training 
        train_fn(loader, disc, gen, 
                    opt_disc, opt_gen, loss_fn, 
                    scaler, DEVICE, 
                    cfg.hyperparams.BATCH_SIZE, 
                    cfg.hyperparams.NUM_EPOCHS, 
                    epoch
                )        




#%% 
if __name__ == '__main__':
    main() 