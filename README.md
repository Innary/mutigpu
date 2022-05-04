# DDP
## dataset  

use:**cifar10**  

---

you can use:     

`python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port 29501 train.py`  
to train model on 4 gpu