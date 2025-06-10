import torch
syncnet.load_state_dict(torch.load(args.syncnet_checkpoint_path, map_location='cpu'))
print(checkpoint.keys())
