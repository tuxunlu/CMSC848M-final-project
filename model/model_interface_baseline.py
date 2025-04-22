import inspect
import torch
import importlib
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from typing import Callable, Dict, Tuple
from .loss.translation_loss import translation_loss


class ModelInterfaceBaseline(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.__load_model()
        self.loss_function = self.__configure_loss()

    def forward(self, x):
        return self.model(x)

    # Caution: self.model.train() is invoked
    def training_step(self, batch, batch_idx):
        image, caption = batch
        image_codebook, translated_image_codebook = self(image, caption)
        train_loss = self.loss_function(translated_image_codebook, image_codebook, 'train')

        self.log('train_loss', train_loss, on_step=True, on_epoch=False, prog_bar=True)
        # Replace the following with evaluations like BLEU score
        # self.log('train_acc', correct_num / len(out_digit), on_step=True, on_epoch=False, prog_bar=True)

        return train_loss

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def validation_step(self, batch, batch_idx):
        image, caption = batch
        image_codebook, translated_image_codebook = self(image, caption)
        val_loss = self.loss_function(translated_image_codebook, image_codebook, 'train')

        self.log('val_loss', val_loss, on_step=True, on_epoch=False, prog_bar=True)
        # Replace the following with evaluations like BLEU score
        # self.log('train_acc', correct_num / len(out_digit), on_step=True, on_epoch=False, prog_bar=True)

        return val_loss

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def test_step(self, batch, batch_idx):
        image, caption = batch
        image_codebook, translated_image_codebook = self(image, caption)
        test_loss = self.loss_function(translated_image_codebook, image_codebook, 'train')

        self.log('train_loss', test_loss, on_step=True, on_epoch=False, prog_bar=True)
        # Replace the following with evaluations like BLEU score
        # self.log('train_acc', correct_num / len(out_digit), on_step=True, on_epoch=False, prog_bar=True)

        return test_loss

    # When there are multiple optimizers, modify this function to fit in your needs
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay)
        )

        # No learning rate scheduler, just return the optimizer
        if self.hparams.lr_scheduler is None:
            return [optimizer]

        # Return tuple of optimizer and learning rate scheduler
        if self.hparams.lr_scheduler == 'step':
            scheduler = lrs.StepLR(
                optimizer,
                step_size=self.hparams.lr_decay_epochs,
                gamma=self.hparams.lr_decay_rate
            )
        elif self.hparams.lr_scheduler == 'cosine':
            scheduler = lrs.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.lr_decay_epochs,
                eta_min=self.hparams.lr_decay_min_lr
            )
        else:
            raise ValueError('Invalid lr_scheduler type!')
        return [optimizer], [scheduler]

    def __calculate_loss_and_log(self, inputs, labels, loss_dict: Dict[str, Tuple[float, Callable]], stage: str):
        raw_loss_list = [func(inputs, labels) for _, func in loss_dict.values()]
        weighted_loss = [weight * raw_loss for (weight, _), raw_loss in zip(loss_dict.values(), raw_loss_list)]
        for name, raw_loss in zip(loss_dict.keys(), raw_loss_list):
            self.log(f'{stage}_{name}', raw_loss.item(), on_step=False, on_epoch=True, prog_bar=False)

        return sum(weighted_loss)

    def __configure_loss(self):
        # User-defined function list. Recommend using `_loss` suffix in loss names.
        user_loss_dict = {
            "translation_loss": translation_loss
        }

        loss_dict = {**user_loss_dict}

        def loss_func(inputs, labels, stage):
            return self.__calculate_loss_and_log(
                inputs=inputs,
                labels=labels,
                loss_dict=loss_dict,
                stage=stage
            )

        return loss_func

    def __load_model(self):
        name = self.hparams.model_class_name
        # Attempt to import the `CamelCase` class name from the `snake_case.py` module. The module should be placed
        # within the same folder as model_interface.py. Always name your model file name as `snake_case.py` and
        # model class name as corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            model_class = getattr(importlib.import_module('.' + name, package=__package__), camel_name)
        except Exception:
            raise ValueError(f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        model = self.__instantiate(model_class)
        if self.hparams.use_compile:
            torch.compile(model)
        return model

    def __instantiate(self, model_class, **other_args):
        # Instantiate a model using the imported class name and parameters from self.hparams dictionary.
        # You can also input any args to overwrite the corresponding value in self.hparams.
        target_args = inspect.getfullargspec(model_class.__init__).args[1:]
        this_args = self.hparams.keys()
        merged_args = {}
        # Only assign arguments that are required in the user-defined torch.nn.Module subclass by their name.
        # You need to define the required arguments in main function.
        for arg in target_args:
            if arg in this_args:
                merged_args[arg] = getattr(self.hparams, arg)

        merged_args.update(other_args)
        return model_class(**merged_args)
