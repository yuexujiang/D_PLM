import torch
import torch.nn.functional as F
import os
import yaml
import shutil
import numpy as np
import logging as log
from scipy.ndimage import zoom
from box import Box
from pathlib import Path
import datetime
from timm import optim
from torch.utils.tensorboard import SummaryWriter
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import ast

def prepare_tensorboard(result_path):
    train_path = os.path.join(result_path, 'train')
    val_path = os.path.join(result_path, 'val')
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(val_path).mkdir(parents=True, exist_ok=True)
    
    train_log_path = os.path.join(train_path, 'tensorboard')
    train_writer = SummaryWriter(train_log_path)
    
    val_log_path = os.path.join(val_path, 'tensorboard')
    val_writer = SummaryWriter(val_log_path)
    
    return train_writer, val_writer


def get_logging(result_path):
    logger = log.getLogger(result_path)
    logger.setLevel(log.INFO)

    fh = log.FileHandler(os.path.join(result_path, "logs.txt"))
    formatter = log.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = log.StreamHandler()
    logger.addHandler(sh)

    return logger


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)

    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)


def prepare_saving_dir(configs, config_file_path):
    """
    Prepare a directory for saving a training results.

    Args:
        configs: A python box object containing the configuration options.

    Returns:
        str: The path to the directory where the results will be saved.
    """
    # Create a unique identifier for the run based on the current time.
    run_id = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    # Create the result directory and the checkpoint subdirectory.
    #result_path = os.path.abspath(os.path.join(configs.result_path, run_id))
    result_path = os.path.abspath(configs.result_path) #on 10/10/2024 for each access and load the best_model for the same results_path!
    
    checkpoint_path = os.path.join(result_path, 'checkpoints')
    figures_path = os.path.join(result_path, 'figures')
    Path(result_path).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path(figures_path).mkdir(parents=True, exist_ok=True)

    # Copy the config file to the result directory.
    shutil.copy(config_file_path, result_path)

    return result_path, checkpoint_path


def test_gpu_cuda():
    print('Testing gpu and cuda:')
    print('\tcuda is available:', torch.cuda.is_available())
    print('\tdevice count:', torch.cuda.device_count())
    print('\tcurrent device:', torch.cuda.current_device())
    print(f'\tdevice:', torch.cuda.device(0))
    print('\tdevice name:', torch.cuda.get_device_name(), end='\n\n')


def load_configs(config, args=None):
    """
        Load the configuration file and convert the necessary values to floats.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            The updated configuration dictionary with float values.
        """

    # Convert the dictionary to a Box object for easier access to the values.
    tree_config = Box(config)

    # Convert the necessary values to floats.
    tree_config.optimizer.lr_seq = float(tree_config.optimizer.lr_seq)
    tree_config.optimizer.lr_struct = float(tree_config.optimizer.lr_struct)
    tree_config.optimizer.decay.min_lr_struct = float(tree_config.optimizer.decay.min_lr_struct)
    tree_config.optimizer.decay.min_lr_seq = float(tree_config.optimizer.decay.min_lr_seq)
    tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
    tree_config.optimizer.eps = float(tree_config.optimizer.eps)
    tree_config.train_settings.temperature = float(tree_config.train_settings.temperature)
    tree_config.optimizer.beta_1 = float(tree_config.optimizer.beta_1)
    tree_config.optimizer.beta_2 = float(tree_config.optimizer.beta_2)
    tree_config.model.esm_encoder.lora.dropout = float(tree_config.model.esm_encoder.lora.dropout)
    tree_config.model.struct_encoder.lora.dropout = float(tree_config.model.struct_encoder.lora.dropout)

    # overwrite parameters if set through commandline
    if args is not None:
        if args.result_path:
            tree_config.result_path = args.result_path

        if args.resume_path:
            tree_config.resume.resume_path = args.resume_path

        if args.num_end_adapter_layers:
            tree_config.encoder.adapter_h.num_end_adapter_layers = int(args.num_end_adapter_layers)

        if args.module_type:
            tree_config.encoder.adapter_h.module_type = args.module_type
        
        if args.restart_optimizer:
           tree_config.resume.restart_optimizer = args.restart_optimizer

    
    #set configs value to default if doesn't have the attr
    if not hasattr(tree_config.model.struct_encoder, "use_seq"):
        tree_config.model.struct_encoder.use_seq = None
        tree_config.model.struct_encoder.use_seq.enable = False
        tree_config.model.struct_encoder.use_seq.seq_embed_mode = "embedding"
        tree_config.model.struct_encoder.use_seq.seq_embed_dim = 20
    
    if not hasattr(tree_config.model.struct_encoder,"top_k"):
       tree_config.model.struct_encoder.top_k = 30 #default
    
    if not hasattr(tree_config.model.struct_encoder,"gvp_num_layers"):
       tree_config.model.struct_encoder.gvp_num_layers = 3 #default
    
    if not hasattr(tree_config.model.struct_encoder,"use_rotary_embeddings"): #configs also have num_rbf and num_positional_embeddings
        tree_config.model.struct_encoder.use_rotary_embeddings=False
    
    if not hasattr(tree_config.model.struct_encoder,"rotary_mode"):
        tree_config.model.struct_encoder.rotary_mode=1
    
    if not hasattr(tree_config.model.struct_encoder,"use_foldseek"): #configs also have num_rbf and num_positional_embeddings
        tree_config.model.struct_encoder.use_foldseek=False
    
    if not hasattr(tree_config.model.struct_encoder,"use_foldseek_vector"): #configs also have num_rbf and num_positional_embeddings
        tree_config.model.struct_encoder.use_foldseek_vector=False
    
    if not hasattr(tree_config.model.struct_encoder,"num_rbf"):
       tree_config.model.struct_encoder.num_rbf = 16 #default
    
    if not hasattr(tree_config.model.struct_encoder,"num_positional_embeddings"):
       tree_config.model.struct_encoder.num_positional_embeddings = 16 #default
    
    if not hasattr(tree_config.model.struct_encoder,"node_h_dim"):
       tree_config.model.struct_encoder.node_h_dim = (100,32) #default
    else:
       tree_config.model.struct_encoder.node_h_dim = ast.literal_eval(tree_config.model.struct_encoder.node_h_dim)
    
    if not hasattr(tree_config.model.struct_encoder,"edge_h_dim"):
       tree_config.model.struct_encoder.edge_h_dim = (32,1) #default
    else:
       tree_config.model.struct_encoder.edge_h_dim = ast.literal_eval(tree_config.model.struct_encoder.edge_h_dim)
    
    
    
    return tree_config


def load_checkpoints(simclr, configs, optimizer_seq,optimizer_struct,scheduler_seq,scheduler_struct,
                     logging,resume_path,restart_optimizer=False):
    start_step=0
    best_score = None
    assert os.path.exists(resume_path), (f'resume_path not exits')
    checkpoint = torch.load(resume_path)  # , map_location=lambda storage, loc: storage)
    print(f"load checkpoints from {resume_path}")
    logging.info(f"load checkpoints from {resume_path}")
    
    if "state_dict2" in checkpoint:
         simclr.model_struct.load_state_dict(checkpoint['state_dict2'])
    
    if "classification_layer" in checkpoint and hasattr(simclr,'classification_layer'):
         simclr.classification_layer.load_state_dict(checkpoint['classification_layer'])
    
    if "struct_decoder" in checkpoint and hasattr(simclr,'struct_decoder'):
         simclr.struct_decoder.load_state_dict(checkpoint['struct_decoder'])
    
    if "transformer_layer" in checkpoint and hasattr(simclr,'transformer_layer'):
         simclr.transformer_layer.load_state_dict(checkpoint['transformer_layer'])
    
    if "pos_embed_decoder" in checkpoint and hasattr(simclr,'pos_embed_decoder'):
         simclr.pos_embed_decoder.load(checkpoint['pos_embed_decoder'])
    
    if "state_dict1" in checkpoint: 
        # to load old checkpoints that saved adapter_layer_dict as adapter_layer.
        from collections import OrderedDict
        if np.sum(["adapter_layer_dict" in key for key in checkpoint[
            'state_dict1'].keys()]) == 0:  # using old checkpoints, need to rename the adapter_layer into adapter_layer_dict.adapter_0
            new_ordered_dict = OrderedDict()
            for key, value in checkpoint['state_dict1'].items():
                if "adapter_layer_dict" not in key:
                    new_key = key.replace('adapter_layer', 'adapter_layer_dict.adapter_0')
                    new_ordered_dict[new_key] = value
                else:
                    new_ordered_dict[key] = value
            
            simclr.model_seq.load_state_dict(new_ordered_dict)
        else:  # new checkpoints with new code, that can be loaded directly.
            simclr.model_seq.load_state_dict(checkpoint['state_dict1'])
    
    if 'step' in checkpoint:
        if not restart_optimizer:
            if 'optimizer_struct' in checkpoint and "scheduler_struct" in checkpoint:
                optimizer_struct.load_state_dict(checkpoint['optimizer_struct'])
                logging.info('optimizer_struct is loaded to resume training!')
                scheduler_struct.load_state_dict(checkpoint['scheduler_struct'])
                logging.info('scheduler_struct is loaded to resume training!')
            if 'optimizer_seq' in checkpoint and 'scheduler_seq' in checkpoint:
                optimizer_seq.load_state_dict(checkpoint['optimizer_seq'])
                logging.info('optimizer_seq is loaded to resume training!')
                scheduler_seq.load_state_dict(checkpoint['scheduler_seq'])
                logging.info('scheduler_seq is loaded to resume training!')
            
            start_step = checkpoint['step'] + 1
            if 'best_score' in checkpoint:
                best_score = checkpoint['best_score']
    
    
    if hasattr(configs.model, 'memory_banck'):
        if configs.model.memory_banck.enable:
            # """
            if "seq_queue" in checkpoint:
                simclr.register_buffer('seq_queue', checkpoint['seq_queue'])
                simclr.register_buffer('seq_queue_ptr', checkpoint['seq_queue_ptr'])
                simclr.register_buffer('struct_queue', checkpoint['struct_queue'])
                simclr.register_buffer('struct_queue_ptr', checkpoint['struct_queue_ptr'])
                # simclr.seq_queue=simclr.seq_queue.to('cpu')
                # simclr.seq_queue_ptr=simclr.seq_queue_ptr.to('cpu')
                # simclr.struct_queue=simclr.struct_queue.to('cpu')
                # simclr.struct_queue_ptr=simclr.struct_queue_ptr.to('cpu')
                """
            simclr.seq_queue.load_state_dict(checkpoint['seq_queue'])
            simclr.seq_queue_ptr.load_state_dict(checkpoint['seq_queue_ptr'])
            simclr.struct_queue.load_state_dict(checkpoint['struct_queue'])
            simclr.struct_queue_ptr.load_state_dict(checkpoint['struct_queue_ptr'])
            """

    return simclr,start_step,best_score



def save_checkpoints(optimizer_x, optimizer_seq, result_path, simclr, n_steps, logging, epoch):
    checkpoint_name = f'checkpoint_{n_steps:07d}.pth'
    save_path = os.path.join(result_path, 'checkpoints', checkpoint_name)

    save_checkpoint({
        'epoch': epoch,
        'step': n_steps,
        'state_dict1': simclr.model_seq.state_dict(),
        'state_x': simclr.model_x.state_dict(),
        'optimizer_x': optimizer_x.state_dict(),
        'optimizer_seq': optimizer_seq.state_dict(),
    }, is_best=False, filename=save_path)
    logging.info(f"Model checkpoint and metadata have been saved at {save_path}")


def save_struct_checkpoints(optimizer_struct, result_path, model, n_steps, logging, epoch,checkpoint_name=None):
    if checkpoint_name is None:
       checkpoint_name = f'checkpoint_{n_steps:07d}.pth'
    
    save_path = os.path.join(result_path, 'checkpoints', checkpoint_name)
    
    save_checkpoint({
        'epoch': epoch,
        'step': n_steps,
        'state_dict2': model.model_struct.state_dict(),
        'classification_layer':model.classification_layer.state_dict(),
        'optimizer_struct': optimizer_struct.state_dict(),
    }, is_best=False, filename=save_path)
    logging.info(f"Model checkpoint and metadata for structure have been saved at {save_path}")

def save_contrastive_struct_checkpoints(optimizer_struct, scheduler_struct,result_path, model, n_steps, logging,epoch,best_score,checkpoint_name=None):
    if checkpoint_name is None:
       checkpoint_name = f'checkpoint_{n_steps:07d}.pth'
    
    save_path = os.path.join(result_path, 'checkpoints', checkpoint_name)
    
    save_checkpoint({
        'epoch': epoch,
        'step': n_steps,
        'best_score': best_score,
        'state_dict2': model.model_struct.state_dict(),
        'wholemodel':model.state_dict(),
        'optimizer_struct': optimizer_struct.state_dict(),
        'scheduler_struct': scheduler_struct.state_dict(),
    }, is_best=False, filename=save_path)
    logging.info(f"Model checkpoint and metadata for structure have been saved at {save_path}")
    return save_path


def save_structpretrain_checkpoints(optimizer_struct, result_path, model, n_steps, logging, epoch):
    checkpoint_name = f'checkpoint_{n_steps:07d}.pth'
    save_path = os.path.join(result_path, 'checkpoints', checkpoint_name)
    
    save_checkpoint({
        'epoch': epoch,
        'step': n_steps,
        'state_dict2': model.model_struct.state_dict(),
        'pos_embed_decoder':model.pos_embed_decoder,
        'transformer_layer':model.transformer_layer.state_dict(),
        'struct_decoder':model.struct_decoder.state_dict(),
        'optimizer_struct': optimizer_struct.state_dict(),
    }, is_best=False, filename=save_path)
    logging.info(f"Model checkpoint and metadata for structure have been saved at {save_path}")



def load_optimizers(model_seq, model_x, logging, configs):
    optimizer_seq=None
    optimizer_x=None
    if configs.optimizer.name.lower() == 'adabelief':
        if model_seq is not None:
            optimizer_seq = optim.AdaBelief(model_seq.parameters(), lr=configs.optimizer.lr_seq, eps=configs.optimizer.eps,
                                        decoupled_decay=True,
                                        weight_decay=configs.optimizer.weight_decay, rectify=False)
        if model_x is not None and configs.model.X_module=="structure":
            optimizer_x = optim.AdaBelief(model_x.parameters(), lr=configs.optimizer.lr_struct,
                                           eps=configs.optimizer.eps,
                                           decoupled_decay=True,
                                           weight_decay=configs.optimizer.weight_decay, rectify=False)
        if model_x is not None and configs.model.X_module=="MD":
            optimizer_x = optim.AdaBelief(model_x.parameters(), lr=configs.optimizer.lr_MD,
                                           eps=configs.optimizer.eps,
                                           decoupled_decay=True,
                                           weight_decay=configs.optimizer.weight_decay, rectify=False)
    elif configs.optimizer.name.lower() == 'adam':
        if model_seq is not None:
          optimizer_seq = torch.optim.AdamW(
            model_seq.parameters(), lr=float(configs.optimizer.lr_seq),
            betas=(configs.optimizer.beta_1, configs.optimizer.beta_2),
            weight_decay=float(configs.optimizer.weight_decay),
            eps=float(configs.optimizer.eps)
          )
        if model_x is not None and configs.model.X_module=="structure":
          optimizer_x = torch.optim.AdamW(
            model_x.parameters(), lr=float(configs.optimizer.lr_MD),
            betas=(configs.optimizer.beta_1, configs.optimizer.beta_2),
            weight_decay=float(configs.optimizer.weight_decay),
            eps=float(configs.optimizer.eps)
          )
    elif configs.optimizer.name.lower() == 'sgd':
        logging.info('use sgd optimizer')
        if model_x is not None and configs.model.X_module=="structure":
            optimizer_x = torch.optim.SGD(model_x.parameters(), lr=float(configs.optimizer.lr_struct),
                                           momentum=0.9, dampening=0,
                                           weight_decay=float(configs.optimizer.weight_decay))
        if model_x is not None and configs.model.X_module=="MD":
            optimizer_x = torch.optim.SGD(model_x.parameters(), lr=float(configs.optimizer.lr_MD),
                                           momentum=0.9, dampening=0,
                                           weight_decay=float(configs.optimizer.weight_decay))
        if model_seq is not None:
           optimizer_seq = torch.optim.SGD(model_seq.parameters(), lr=float(configs.optimizer.lr_seq), momentum=0.9,
                                        dampening=0, weight_decay=float(configs.optimizer.weight_decay))

    else:
        raise ValueError('wrong optimizer')
    return optimizer_seq, optimizer_x


def prepare_optimizer(model_seq, model_x, logging, configs):
    logging.info("prepare the optimizers")
    optimizer_seq, optimizer_x = load_optimizers(model_seq, model_x, logging, configs)
    scheduler_seq, scheduler_x= None,None
    #print(optimizer_seq.param_groups[0].keys())
    #print(optimizer_struct.param_groups[0].keys())

    logging.info("prepare the schedulers")
    # if configs.optimizer.name.lower() != 'sgd':
    if model_x is not None and configs.model.X_module=="structure":
      scheduler_x = CosineAnnealingWarmupRestarts(
        optimizer_x,
        first_cycle_steps=configs.optimizer.decay.first_cycle_steps,
        cycle_mult=1.0,
        max_lr=configs.optimizer.lr_struct,
        min_lr=configs.optimizer.decay.min_lr_struct,
        warmup_steps=configs.optimizer.decay.warmup,
        gamma=configs.optimizer.decay.gamma)
    if model_x is not None and configs.model.X_module=="MD":
      scheduler_x = CosineAnnealingWarmupRestarts(
        optimizer_x,
        first_cycle_steps=configs.optimizer.decay.first_cycle_steps,
        cycle_mult=1.0,
        max_lr=configs.optimizer.lr_MD,
        min_lr=configs.optimizer.decay.min_lr_MD,
        warmup_steps=configs.optimizer.decay.warmup,
        gamma=configs.optimizer.decay.gamma)
    if model_seq is not None:
      scheduler_seq = CosineAnnealingWarmupRestarts(
        optimizer_seq,
        first_cycle_steps=configs.optimizer.decay.first_cycle_steps,
        cycle_mult=1.0,
        max_lr=configs.optimizer.lr_seq,
        min_lr=configs.optimizer.decay.min_lr_seq,
        warmup_steps=configs.optimizer.decay.warmup,
        gamma=configs.optimizer.decay.gamma)
    # else:
    #    logging.info("use sgd and old lr_scheduler")
    #    scheduler_struct = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_struct, configs.optimizer.decay.T0, configs.optimizer.decay.Tmult)
    #    scheduler_seq = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_seq, configs.optimizer.decay.T0, configs.optimizer.decay.Tmult)

    return scheduler_seq, scheduler_x, optimizer_seq, optimizer_x


def plot_learning_rate(scheduler, optimizer, num_steps, filename):
    import matplotlib.pyplot as plt
    # Collect learning rates
    lrs = []

    # Simulate a training loop
    for epoch in range(num_steps):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    # Plot the learning rates
    plt.plot(lrs)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    # plt.show()
    plt.savefig(filename)


def Image_resize(matrix, max_len):
    bsz = len(matrix)
    new_size = [max_len, max_len]
    pad_contact = np.zeros((bsz, max_len, max_len), dtype=float)
    for index in range(bsz):
        scale_factors = (new_size[0] / matrix[index].shape[0], new_size[1] / matrix[index].shape[1])
        pad_contact[index] = zoom(matrix[index], zoom=scale_factors,
                                  order=0)  # order=0 for nearest-neighbor, order=3 for bicubic order=1 to perform bilinear interpolation, which smoothly scales the matrix content to the desired size.

    return pad_contact


def pad_concatmap(matrix, max_len, pad_value=255):
    bsz = len(matrix)
    pad_contact = np.zeros((bsz, max_len, max_len), dtype=float)
    mask_matrix = np.full((bsz, max_len, max_len), False, dtype=bool)
    for i in range(bsz):
        leng = len(matrix[i])
        if leng >= max_len:
            # print(i)
            pad_contact[i, :, :] = matrix[i][:max_len,
                                   :max_len]  # we trim the contact map 2D matrix if it's dimension > max_len
            mask_matrix[i, :, :] = True
        else:
            # print(i)
            # print("len < 224")
            pad_len = max_len - leng
            pad_contact[i, :, :] = np.pad(matrix[i], [(0, pad_len), (0, pad_len)], mode='constant',
                                          constant_values=pad_value)
            mask_matrix[i, :leng, :leng] = True
            # print(mask_matrix[i].shape)
            # print(mask_matrix[i])

    return pad_contact, mask_matrix


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def residue_batch_sample_old(residue_struct, residue_seq, plddt_residue, sample_size):
    sample_size = np.min([len(residue_struct), sample_size])
    random_indices = np.random.choice(len(residue_struct), sample_size, replace=False)
    residue_struct = residue_struct[random_indices, :]
    residue_seq = residue_seq[random_indices, :]
    plddt_residue = plddt_residue[random_indices]
    return residue_struct, residue_seq, plddt_residue


def residue_batch_sample(residue_struct, residue_seq, plddt_residue, sample_size, accelerator):
    # Adjust sample size if it's larger than the number of residues
    sample_size = min(residue_struct.size(0), sample_size)

    # Randomly select indices without replacement
    random_indices = torch.randperm(len(residue_struct), device=accelerator.device)[:sample_size]

    # Index the tensors to get the sampled batch
    residue_struct = residue_struct[random_indices, :]
    residue_seq = residue_seq[random_indices, :]
    plddt_residue = plddt_residue[random_indices]

    return residue_struct, residue_seq, plddt_residue
