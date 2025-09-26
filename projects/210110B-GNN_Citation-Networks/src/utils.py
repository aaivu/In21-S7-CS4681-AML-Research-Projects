import os
import glob
import paddle
from pgl.utils.logger import log


def _create_if_not_exist(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)


def save_model(model_path, model, epoch, opt=None, lr_scheduler=None):
    """Save model checkpoint"""
    _create_if_not_exist(model_path)
    
    model_file = os.path.join(model_path, f'model_epoch_{epoch}.pdparams')
    
    if hasattr(model, '_layers'):
        # DataParallel model
        paddle.save(model._layers.state_dict(), model_file)
    else:
        paddle.save(model.state_dict(), model_file)
    
    log.info(f"Model saved to {model_file}")
    
    if opt is not None:
        opt_file = os.path.join(model_path, f'opt_epoch_{epoch}.pdopt')
        paddle.save(opt.state_dict(), opt_file)
        
    if lr_scheduler is not None:
        scheduler_file = os.path.join(model_path, f'scheduler_epoch_{epoch}.pdlr')
        paddle.save(lr_scheduler.state_dict(), scheduler_file)


def load_model(model_path, model, opt=None, lr_scheduler=None, epoch=None):
    """Load model checkpoint"""
    if not os.path.exists(model_path):
        log.info(f"Model path {model_path} not found, starting from scratch")
        return
    
    # Find the latest model if epoch not specified
    if epoch is None:
        model_files = glob.glob(os.path.join(model_path, 'model_epoch_*.pdparams'))
        if not model_files:
            log.info(f"No model files found in {model_path}")
            return
        
        # Extract epoch numbers and find the latest
        epochs = [int(f.split('_')[-1].split('.')[0]) for f in model_files]
        epoch = max(epochs)
    
    model_file = os.path.join(model_path, f'model_epoch_{epoch}.pdparams')
    
    if os.path.exists(model_file):
        if hasattr(model, '_layers'):
            # DataParallel model
            model._layers.set_state_dict(paddle.load(model_file))
        else:
            model.set_state_dict(paddle.load(model_file))
        log.info(f"Model loaded from {model_file}")
        
        # Load optimizer state
        if opt is not None:
            opt_file = os.path.join(model_path, f'opt_epoch_{epoch}.pdopt')
            if os.path.exists(opt_file):
                opt.set_state_dict(paddle.load(opt_file))
                log.info(f"Optimizer loaded from {opt_file}")
                
        # Load scheduler state
        if lr_scheduler is not None:
            scheduler_file = os.path.join(model_path, f'scheduler_epoch_{epoch}.pdlr')
            if os.path.exists(scheduler_file):
                lr_scheduler.set_state_dict(paddle.load(scheduler_file))
                log.info(f"Scheduler loaded from {scheduler_file}")
    else:
        log.info(f"Model file {model_file} not found")


def infinite_loop(data_loader):
    """Create an infinite loop over the data loader"""
    while True:
        for batch in data_loader:
            yield batch
