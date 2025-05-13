import torch
import torch.nn as nn

from .utils.vqvae import VQVAE
from .utils.transformer import Transformer
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
from .model_interface_baseline import ModelInterfaceBaseline

class Baseline(nn.Module):
    def __init__(
            self,
            # Transformer parameters
            src_vocab_size, 
            tgt_vocab_size, 
            d_model=512, 
            num_heads=8,
            num_encoder_layers=6, 
            num_decoder_layers=6, 
            dim_ff=1024, 
            dropout=0.1, 
            src_max_len=512,
            tgt_max_len=512,
            tgt_start_token_id=1024,
            # VQVAE parameters
            h_dim=128, 
            res_h_dim=128, 
            n_res_layers=4,
            n_embeddings=1024, 
            embedding_dim=128, 
            beta=0.25, 
            save_img_embedding_map=False,
            pretrained_vqvae_path=None,
            pretrained_transformer_path=None,
            downsample_height=4,
            downsample_width=4,
        ):
        super(Baseline, self).__init__()
        if 'pretrained_vqvae_path' == None:
            raise KeyError("Key pretrained_vqvae_path is None!")
        
        self.downsample_height = downsample_height
        self.downsample_width = downsample_width
        self.tgt_start_token_id = tgt_start_token_id

        self.vqvae = self.load_vqvae_model(
            pretrained_vqvae_path, 
            h_dim=h_dim, 
            res_h_dim=res_h_dim, 
            n_res_layers=n_res_layers,
            n_embeddings=n_embeddings, 
            embedding_dim=embedding_dim, 
            beta=beta, 
            save_img_embedding_map=save_img_embedding_map,
            downsample_height=downsample_height,
            downsample_width=downsample_width,
        )
        self.transformer = Transformer(
            src_vocab_size=src_vocab_size, 
            tgt_vocab_size=tgt_vocab_size, 
            d_model=d_model, 
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_ff=dim_ff, 
            dropout=dropout, 
            input_max_len=src_max_len,
            output_max_len=tgt_max_len,
        ) if pretrained_transformer_path == 'None' else self.load_pretrained_transformer(
            pretrained_path=pretrained_transformer_path, 
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size, 
            d_model=d_model, 
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_ff=dim_ff, 
            dropout=dropout, 
            input_max_len=src_max_len,
            output_max_len=tgt_max_len,
        )

        # Freeze parameters of VQVAE for inference only
        for param in self.vqvae.parameters():
            param.requires_grad = False

        self.vqvae.eval()

    def load_vqvae_model(self, pretrained_vqvae_path, **kwargs):
        model = VQVAE(**kwargs)
        checkpoint = torch.load(pretrained_vqvae_path)
        
        # Remove "model." prefix
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("model.", "")  # Remove the prefix
            new_state_dict[new_k] = v

        model.load_state_dict(new_state_dict)
        return model
    
    def load_pretrained_transformer(self, pretrained_path, **kwargs):
        model = Transformer(**kwargs)

        checkpoint = torch.load(pretrained_path)
        
        # Remove "model." prefix
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'transformer' in k:
                new_k = k.replace("model.transformer.", "")
                new_state_dict[new_k] = v

        model.load_state_dict(new_state_dict)
        return model
        
    
    def forward(self, image, caption, src_mask, gen_image):
        B = image.shape[0]
        total_len = int((120 / self.downsample_height) * (160 / self.downsample_width))
        src_mask = src_mask.bool()
        # Create causal mask to prevent attending to future tokens
        tgt_mask = torch.triu(torch.ones(total_len + 1, total_len + 1, device=image.device), diagonal=1).bool()
        tgt_mask = tgt_mask.masked_fill(tgt_mask, float('-inf'))
        with torch.no_grad():
            image_sentence = self.vqvae.encoder_forward(image) # (B, total_len)
            if not gen_image:
                # autoregressive codebook generation
                generated = torch.full((B, 1), self.tgt_start_token_id,
                                        dtype=torch.long, device=image.device)
                logits_seq = []
                for t in range(total_len):
                    # rebuild causal mask for this length
                    tgt_mask = torch.triu(torch.ones(generated.size(1), generated.size(1), device=image.device), 1)
                    tgt_mask = tgt_mask.bool().masked_fill(tgt_mask.bool(), float('-inf'))

                    logits = self.transformer(caption, generated,
                                                src_mask=src_mask, tgt_mask=tgt_mask)
                    # mask out the <start> token id
                    logits[:, :, self.tgt_start_token_id] = -float('inf')

                    next_token = logits[:, -1, :].argmax(-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
                    logits_seq.append(logits[:, -1, :])

                # now image_sentence = the codebook IDs we generated (sans start token)
                pred_image_sentence = generated[:, 1:]             # (B, total_len)
                pred_sentence_prob = torch.stack(logits_seq, dim=1)

                print("pred_image_sentence.shape=", pred_image_sentence.shape)
                print((pred_image_sentence == self.tgt_start_token_id).nonzero())
                print(pred_image_sentence)
                print(image_sentence)
                # save image_sentence as json
                with open('/fs/nexus-scratch/tuxunlu/git/CMSC848M-final-project/image_sentence.json', 'w') as f:
                    f.write(str(image_sentence[0].tolist()))
                # save pred_image_sentence as json
                with open('/fs/nexus-scratch/tuxunlu/git/CMSC848M-final-project/pred_image_sentence.json', 'w') as f:
                    f.write(str(pred_image_sentence[0].tolist()))

            else:
                
                # Add a start token at the front of each image sentence
                start_token = torch.full(
                    (image_sentence.size(0), 1),  # (B, 1)
                    fill_value=self.tgt_start_token_id,
                    dtype=torch.long,
                    device=image_sentence.device
                )
                image_sentence = torch.cat([start_token, image_sentence], dim=1)  # (B, total_len+1)
                pred_sentence_prob = self.transformer(caption, image_sentence, src_mask=src_mask, tgt_mask=tgt_mask)

        pred_image = None
        if not gen_image:
            with torch.no_grad():
                pred_image = self.vqvae.decoder_forward(pred_image_sentence, self.downsample_height, self.downsample_width)
        else:
            pred_image_sentence = pred_sentence_prob.argmax(dim=-1)
            # truncate the <start> token at front
            pred_image_sentence = pred_image_sentence[:, 1:]  # (B, 1, total_len)
            print("pred_image_sentence.shape=", pred_image_sentence.shape)
            pred_image = self.vqvae.decoder_forward(pred_image_sentence, self.downsample_height, self.downsample_width)

        # Save results
        batch_id = 0

        real_image = ModelInterfaceBaseline.denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        save_image(real_image[batch_id].float().div(255.0), os.path.join("/fs/nexus-scratch/tuxunlu/git/CMSC848M-final-project/inference", f'real_image.png'))
        gen_image = ModelInterfaceBaseline.denormalize(pred_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        save_image(gen_image[batch_id].float().div(255.0), os.path.join("/fs/nexus-scratch/tuxunlu/git/CMSC848M-final-project/inference", f'gen_image.png'))

        idxs = image_sentence[batch_id][1:].cpu().numpy()
        grid = idxs.reshape(120 // self.downsample_height, 160 // self.downsample_width)
        print("grid.shape=", grid.shape)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.savefig('/fs/nexus-scratch/tuxunlu/git/CMSC848M-final-project/inference/image_sentence.png')
        

        return image_sentence, pred_sentence_prob, pred_image