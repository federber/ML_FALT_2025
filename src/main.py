from src.config import *
#from src.data import *
from src.model import *
from src.utils import *
from src.train import *

import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    DEVICE = torch.device('cuda')
    print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    DEVICE = torch.device('cpu')
    print("No GPU found, running on CPU")

np.random.seed(42)

