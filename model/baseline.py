import torch
import torch.nn as nn

from .utils.vqvae import VQVAE
from .utils.transformer import Transformer


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
            if gen_image:
                generated = torch.full((B, 1), self.tgt_start_token_id, dtype=torch.long, device=image.device)
                
                for t in range(total_len):
                    print("t=", t)
                    tgt_mask = torch.triu(torch.ones(generated.size(1), generated.size(1), device=image.device), diagonal=1).bool()
                    tgt_mask = tgt_mask.masked_fill(tgt_mask, float('-inf'))

                    logits = self.transformer(caption, generated, src_mask=src_mask, tgt_mask=tgt_mask)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # shape: (B, 1)
                    generated = torch.cat([generated, next_token], dim=1)  # append to sequence

                pred_image_sentence = generated[:, 1:]
                print("pred_image_sentence.shape=", pred_image_sentence.shape)

            else:
                image_sentence = self.vqvae.encoder_forward(image) # (B, total_len)
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
        if gen_image:
            with torch.no_grad():
                pred_image = self.vqvae.decoder_forward(pred_image_sentence, self.downsample_height, self.downsample_width)
        else:
            pred_image_sentence = pred_sentence_prob.argmax(dim=-1)
            # truncate the <start> token at front
            pred_image_sentence = pred_image_sentence[:, 1:]  # (B, 1, total_len)
            print("pred_image_sentence.shape=", pred_image_sentence.shape)
            pred_image = self.vqvae.decoder_forward(pred_image_sentence, self.downsample_height, self.downsample_width)


        return image_sentence, pred_sentence_prob, pred_image