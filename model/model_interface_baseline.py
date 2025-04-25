import inspect
import torch
import importlib
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from typing import Callable, Dict, Tuple
from .loss.translation_loss import translation_loss
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics import MeanMetric


class ModelInterfaceBaseline(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.__load_model()
        self.loss_function = self.__configure_loss()

        # Metrics
        self.bleu = BLEUScore()
        self.fid = FrechetInceptionDistance()
        self.psnr = PeakSignalNoiseRatio()
        self.clip = CLIPScore()
        self.test_perplexity_avg = MeanMetric()
        self.test_bleu_avg = MeanMetric()

    def forward(self, image, caption, mask, gen_image):
        return self.model(image, caption, mask, gen_image=gen_image)

    def training_step(self, batch, batch_idx):
        image, caption, mask = batch
        image_codebook, translated_image_codebook, _ = self(image, caption, mask, gen_image=False)
        train_loss = self.loss_function(translated_image_codebook, image_codebook, 'train')
        self.log('train_loss', train_loss, on_step=True, on_epoch=False, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        image, caption, mask = batch
        image_codebook, translated_image_codebook, _ = self(image, caption, mask, gen_image=True)
        val_loss = self.loss_function(translated_image_codebook, image_codebook, 'val')
        self.log('val_loss', val_loss, on_step=True, on_epoch=False, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        image, caption, mask = batch
        image_codebook, translated_image_codebook, gen_image = self(image, caption, mask, gen_image=True)
        test_loss = self.loss_function(translated_image_codebook, image_codebook, 'test')
        self.log('test_loss', test_loss, on_step=True, on_epoch=False, prog_bar=True)

        # Perplexity
        perplexity = torch.exp(test_loss)
        self.test_perplexity_avg.update(perplexity)

        # BLEU score
        preds = translated_image_codebook.tolist()
        refs = image_codebook.tolist()
        bleu_score = self.bleu(preds, refs)
        self.test_bleu_avg.update(bleu_score)

        # FID update
        self.fid.update(gen_image, image)

        # PSNR update
        self.psnr.update(gen_image, image)

        # CLIP score update: compares image and caption strings
        # caption should be a list of strings per batch element
        self.clip.update(gen_image, caption)

        return {'test_loss': test_loss}

    def test_epoch_end(self, outputs):
        # Compute and log average perplexity
        avg_perplexity = self.test_perplexity_avg.compute()
        self.log('perplexity', avg_perplexity, on_epoch=True, prog_bar=True)
        self.test_perplexity_avg.reset()

        # Compute and log average BLEU score
        avg_bleu = self.test_bleu_avg.compute()
        self.log('bleu', avg_bleu, on_epoch=True, prog_bar=True)
        self.test_bleu_avg.reset()

        # Compute and log FID
        fid_value = self.fid.compute()
        self.log('fid', fid_value, on_epoch=True, prog_bar=True)
        self.fid.reset()

        # Compute and log PSNR
        psnr_value = self.psnr.compute()
        self.log('psnr', psnr_value, on_epoch=True, prog_bar=True)
        self.psnr.reset()

        # Compute and log CLIP Score
        clip_value = self.clip.compute()
        self.log('clip_score', clip_value, on_epoch=True, prog_bar=True)
        self.clip.reset()

        return {'perplexity': avg_perplexity, 'bleu': avg_bleu, 'fid': fid_value, 'psnr': psnr_value, 'clip_score': clip_value}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay)
        )
        if self.hparams.lr_scheduler is None:
            return [optimizer]
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

    def __configure_loss(self):
        def loss_func(inputs, labels, stage):
            loss = translation_loss(inputs, labels, pad_token_id=99999)
            self.log(f'{stage}_translation_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=False)
            return loss
        return loss_func

    def __load_model(self):
        name = self.hparams.model_class_name
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
        target_args = inspect.getfullargspec(model_class.__init__).args[1:]
        merged_args = {arg: getattr(self.hparams, arg) for arg in target_args if arg in self.hparams}
        merged_args.update(other_args)
        return model_class(**merged_args)
