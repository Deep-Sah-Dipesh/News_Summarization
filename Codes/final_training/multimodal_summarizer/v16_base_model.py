import torch
import torch.nn as nn
from transformers import MBartForConditionalGeneration, ViTModel
# --- FIX: Added import for BaseModelOutput ---
from transformers.modeling_outputs import BaseModelOutput

class MultimodalSummarizerV16_Base(nn.Module):
    """
    Fuses a Vision Transformer (ViT) with the BASE mBART model.
    """
    def __init__(self, vit_model_name="google/vit-base-patch16-224-in21k", mbart_model_name="facebook/mbart-large-50"):
        super().__init__()

        print(f"Loading ViT from: {vit_model_name}")
        self.vit = ViTModel.from_pretrained(vit_model_name, use_safetensors=True)
        
        print(f"Loading BASE mBART from: {mbart_model_name}")
        self.mbart = MBartForConditionalGeneration.from_pretrained(mbart_model_name, use_safetensors=True)

        vit_config = self.vit.config
        mbart_config = self.mbart.config

        self.projection_layer = nn.Linear(vit_config.hidden_size, mbart_config.d_model)

        print("Freezing ViT and mBART Encoder weights for Stage 1 training...")
        for param in self.vit.parameters():
            param.requires_grad = False
            
        for param in self.mbart.model.encoder.parameters():
            param.requires_grad = False
            
        print("Freezing mBART Decoder (except cross-attention)...")
        for name, param in self.mbart.model.decoder.named_parameters():
            if "encoder_attn" not in name:
                param.requires_grad = False

    def forward(
        self,
        article_input_ids,
        article_attention_mask,
        image_pixel_values,
        labels=None
    ):
        with torch.no_grad():
            image_embeds = self.vit(
                pixel_values=image_pixel_values
            ).last_hidden_state

            text_embeds = self.mbart.model.encoder(
                input_ids=article_input_ids,
                attention_mask=article_attention_mask
            ).last_hidden_state

        projected_image_embeds = self.projection_layer(image_embeds)
        
        image_attention_mask = torch.ones(
            projected_image_embeds.shape[:2], 
            dtype=torch.long, 
            device=projected_image_embeds.device
        )

        combined_embeds = torch.cat([text_embeds, projected_image_embeds], dim=1)
        combined_attention_mask = torch.cat([article_attention_mask, image_attention_mask], dim=1)

        outputs = self.mbart(
            # This is correct for the forward pass
            encoder_outputs=(combined_embeds,),
            attention_mask=combined_attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate(self, article_input_ids, article_attention_mask, image_pixel_values, **gen_kwargs):
        """
        Custom generate function for inference.
        """
        self.eval()
        
        with torch.no_grad():
            image_embeds = self.vit(
                pixel_values=image_pixel_values
            ).last_hidden_state

            text_embeds = self.mbart.model.encoder(
                input_ids=article_input_ids,
                attention_mask=article_attention_mask
            ).last_hidden_state

            projected_image_embeds = self.projection_layer(image_embeds)
            image_attention_mask = torch.ones(
                projected_image_embeds.shape[:2], 
                dtype=torch.long, 
                device=projected_image_embeds.device
            )
            combined_embeds = torch.cat([text_embeds, projected_image_embeds], dim=1)
            combined_attention_mask = torch.cat([article_attention_mask, image_attention_mask], dim=1)

        # --- FIX: Wrap outputs in a BaseModelOutput object ---
        encoder_outputs_object = BaseModelOutput(
            last_hidden_state=combined_embeds
        )
        # ----------------------------------------------------

        generated_ids = self.mbart.generate(
            encoder_outputs=encoder_outputs_object, # <-- Pass the object, not a tuple
            attention_mask=combined_attention_mask,
            **gen_kwargs
        )
        
        return generated_ids