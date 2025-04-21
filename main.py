import os
import datetime

import yaml
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import inspect

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from model import ModelInterface
from data import DataInterface


# For all built-in callback functions, see: https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks
# The following callback functions are commonly used and ready to load based on user setting.
def load_callbacks(config):
    callbacks = []
    # Monitor a metric and stop training when it stops improving
    callbacks.append(plc.EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=10,
        min_delta=0.001
    ))

    # Save the model periodically by monitoring a quantity
    if config['enable_checkpointing']:
        # Best checkpoint
        callbacks.append(plc.ModelCheckpoint(
            every_n_epochs=1,
            monitor='val_acc',
            mode='max',
            filename='best-{epoch:03d}-{val_acc:.5f}',
            save_top_k=1,
            save_last=False,
        ))

        # Epoch checkpoint. Store the model of latest epoch
        callbacks.append(plc.ModelCheckpoint(
            every_n_epochs=1,
            filename='latest-{epoch:03d}-{val_acc:.5f}',
            monitor=None,
        ))

    # Monitor learning rate decay
    callbacks.append(plc.LearningRateMonitor(
        logging_interval='epoch'
    ))

    # Generates a summary of all layers in a LightningModule based on max_depth.
    """
    callbacks.append(plc.ModelSummary(
        max_depth=1
    ))
    """

    # Change gradient accumulation factor according to scheduling.
    # Only consider using this when batch_size does not fit into current hardware environment.
    """
    callbacks.append(plc.GradientAccumulationScheduler(
        # From epoch 5, it starts accumulating every 4 batches. Here we have 4 instead of 5 because epoch (key) should be zero-indexed.
        scheduling={4: 4}
    ))
    """

    return callbacks


def get_checkpoint_path(config):
    # Check if resuming from a manual checkpoint or last checkpoint
    resume_from_manual_checkpoint = config.get('resume_from_manual_checkpoint', None)
    resume_from_last_checkpoint = config.get('resume_from_last_checkpoint', None)
    checkpoint_directory = None
    checkpoint_file_path = None

    if resume_from_manual_checkpoint:
        # After truncating the path, PL will automatically append a new version under the parent folder(which has format: {timestamp}-{experiment name})
        checkpoint_file_path = resume_from_manual_checkpoint
        truncated_path = resume_from_manual_checkpoint
        for i in range(2):
            truncated_path = truncated_path[:truncated_path.rfind(os.path.sep)]
        # Use the same logging directory as the checkpoint
        checkpoint_directory = os.path.dirname(truncated_path)
    # Find the latest folder with latest version
    elif resume_from_last_checkpoint:
        # Find the latest log folder
        current_path = config['log_dir']
        log_dirs = os.listdir(config['log_dir'])
        log_dirs.sort(reverse=True)
        if len(log_dirs) == 0 or len(os.listdir(os.path.join(current_path, log_dirs[0]))) == 0:
            print(f"Warning: resume_from_last_checkpoint was set to True but no checkpoint found at: {current_path}. Launch a new training...")
            return None, None
        
        # Find the latest version folder
        checkpoint_directory = os.path.join(current_path, log_dirs[0])
        version_dirs = os.listdir(checkpoint_directory)
        version_dirs.sort(reverse=True)
        checkpoint_file_path = os.path.join(checkpoint_directory, version_dirs[0], 'checkpoints')
        if not any(s.startswith('latest') and s.endswith('.ckpt') for s in os.listdir(checkpoint_file_path)):
            print(f"Warning: resume_from_last_checkpoint was set to True but no checkpoint file found at: {checkpoint_file_path}. Launch a new training...")
            return None, None

        # Find the latest checkpoint file with largest epoch
        ckpt_files = list(filter(lambda s: s.startswith('latest') and s.endswith('.ckpt'), os.listdir(checkpoint_file_path)))
        ckpt_files.sort(reverse=True)
        checkpoint_file_path = os.path.join(checkpoint_file_path, ckpt_files[0])

    return checkpoint_directory, checkpoint_file_path


def main(config):
    # Set random seed
    pl.seed_everything(config['seed'])

    # Instantiate model and data module
    data_module = DataInterface(**config)
    model_module = ModelInterface(**config)         

    checkpoint_directory, checkpoint_file_path = get_checkpoint_path(config=config) if config['enable_checkpointing'] else (None, None)

    # Caution: the final checkpoint directory depends on the logger path
    # Versions are only used for consecutive checkpointing. If a checkpoint starts from version_n, then new checkpoint directory will be version_{n+1}
    # When the loaded checkpoint reached max_epoch, the new training will stop immediately
    if checkpoint_directory is not None:
        logger = TensorBoardLogger(save_dir='.', name=checkpoint_directory)
    else:
        # Create a new logging directory
        print("Training from scratch...")
        log_dir_name_with_time = os.path.join(config['log_dir'], datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S"))
        logger = TensorBoardLogger(save_dir='.', name=f"{log_dir_name_with_time}-{config['experiment_name']}")
    
    config['logger'] = logger

    # Load callback functions for Trainer
    config['callbacks'] = load_callbacks(config=config)

    # Add resume_from_checkpoint to the trainer initialization
    signature = inspect.signature(Trainer.__init__)
    filtered_trainer_keywords = {}
    for arg in list(signature.parameters.keys()):
        if arg in config:
            filtered_trainer_keywords[arg] = config[arg]

    # Instantiate the Trainer object
    trainer = Trainer(**filtered_trainer_keywords)

    # Launch the training
    trainer.fit(model=model_module, datamodule=data_module, ckpt_path=checkpoint_file_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_path', default=os.path.join(os.getcwd(), 'config', 'config.yaml'), type=str, required=False,
                        help='Path of config file')
    parser.add_argument('--resume_from_last_checkpoint', default=None, type=bool, required=False, 
                    help='Automatically find the log folder with latest timestamp and latest version, and load `latest-...`.ckpt model')
    parser.add_argument('--resume_from_manual_checkpoint', default=None, type=str, required=False,
                    help='Manually designate the path to a checkpoint file(.ckpt) to resume training from.')

    # Parse arguments(set attributes for sys.args using above arguments)
    args = parser.parse_args()

    # Validate config file and dataset directory
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f'No config file found at {args.config_path}!')

    # Load .yaml config file as python dict
    with open(args.config_path) as f:
        config_dict = yaml.safe_load(f)

    # Convert config dict keys to lowercase
    config_dict['resume_from_manual_checkpoint'] = args.resume_from_manual_checkpoint
    config_dict['resume_from_last_checkpoint'] = args.resume_from_last_checkpoint
    config_dict = dict([(k.lower(), v) for k, v in config_dict.items()])

    # Activate main function
    main(config_dict)
