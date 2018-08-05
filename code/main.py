import torch

import utility
import data
import model
import loss

import sys, os

## pass commandline to main.py if it is inside hydrogen
## using the default options in `option.py`
if 'ipykernel_launcher.py' in sys.argv[0] :
    cmd = "main.py --dataDir /home2/data --ext png"
    sys.argv = cmd.split()
    
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

