import torch
import os
import io
import time


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name=''):
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Generate timestamp
    name = f"{args.dataset}_{args.lr_main}_{args.d_vh}_{args.d_vout}"  # Construct filename

    if not os.path.exists('pre_trained_models'):
        os.mkdir('pre_trained_models')

    torch.save(model.state_dict(), f'pre_trained_models/{name}.pt')


def load_model(args, name=''):
    # name = save_load_name(args, name)
    name = f"{args.dataset}_{args.lr_main}_{args.vh}_{args.vout}"
    with open(f'pre_trained_models/{name}.pt', 'rb') as f:
        buffer = io.BytesIO(f.read())
    model = torch.load(buffer)
    return model


def random_shuffle(tensor, dim=0):
    if dim != 0:
        perm = (i for i in range(len(tensor.size())))
        perm[0] = dim
        perm[dim] = 0
        tensor = tensor.permute(perm)
    
    idx = torch.randperm(t.size(0))
    t = tensor[idx]

    if dim != 0:
        t = t.permute(perm)
    
    return t
