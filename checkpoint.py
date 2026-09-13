import torch

def save_checkpoint(path, model, optimizer, step, val_loss, config, train_loader, **kwargs):
    checkpoint = {
        'model': model.state_dict(),  # uncompiled model
        'optimizer': optimizer.state_dict(),
        'step': step,
        'val_loss': val_loss,
        'config': config,  # GPTConfig for rebuilding if needed
        'rng_state': torch.get_rng_state(),
        'loader_pos': (train_loader.current_shard, train_loader.current_position),
        **kwargs  # hellaswag eval or cuda_rng, etc. goes here
    }   
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, device='cpu'):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])

    return ckpt