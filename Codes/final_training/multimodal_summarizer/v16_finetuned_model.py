import torch
import torch.nn as nn
from transformers import MBartForConditionalGeneration, ViTModel
from transformers.modeling_outputs import BaseModelOutput

# This is the path to trained v15 model
V15_MODEL_PATH = r"H:\News_Summarization\codes\final_training\mbart-large-50-cnn-summarizer-v15\final_model"
# This is the BASE mBART model
MBART_MODEL_NAME = "facebook/mbart-large-50"

class MultimodalSummarizerV16_Stage2(nn.Module):
    """
    This model loads the STAGE 1 checkpoint and unfreezes
    the ViT and cross-attention layers for fine-tuning.
    """
    def __init__(self, vit_model_name="google/vit-base-patch16-224-in21k"):
        super().__init__()

        # --- 1. Load Pre-trained Models ---
        print(f"Loading ViT from: {vit_model_name}")
        self.vit = ViTModel.from_pretrained(vit_model_name, use_safetensors=True)
        
        print(f"Loading BASE mBART from: {MBART_MODEL_NAME}")
        self.mbart = MBartForConditionalGeneration.from_pretrained(MBART_MODEL_NAME, use_safetensors=True)

        vit_config = self.vit.config
        mbart_config = self.mbart.config

        # --- 2. Create Projection Layer (Fusion) ---
        self.projection_layer = nn.Linear(vit_config.hidden_size, mbart_config.d_model)

        # --- 3. UNFREEZE LAYERS FOR STAGE 2 ---
        print("Unfreezing ViT and mBART Decoder cross-attention...")
        
        # Unfreeze ViT (Image Encoder)
        for param in self.vit.parameters():
            param.requires_grad = True
            
        # Keep mBART Text Encoder frozen
        for param in self.mbart.model.encoder.parameters():
            param.requires_grad = False
            
        # Unfreeze only the cross-attention and projection layer
        for name, param in self.mbart.model.decoder.named_parameters():
            if "encoder_attn" in name: # 'encoder_attn' is the cross-attention
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Ensure the LM Head and Projection Layer are trainable
        for param in self.mbart.lm_head.parameters():
             param.requires_grad = True
        for param in self.projection_layer.parameters():
             param.requires_grad = True


    def forward(
        self,
        article_input_ids,
        article_attention_mask,
        image_pixel_values,
        labels=None
    ):
        # --- 1. Get Image Embeddings (now trainable) ---
        image_embeds = self.vit(
            pixel_values=image_pixel_values
        ).last_hidden_state

        # --- 2. Get Text Embeddings (still frozen) ---
        with torch.no_grad():
            text_embeds = self.mbart.model.encoder(
                input_ids=article_input_ids,
                attention_mask=article_attention_mask
            ).last_hidden_state

        # --- 3. Fuse Embeddings ---
        projected_image_embeds = self.projection_layer(image_embeds)
        
        image_attention_mask = torch.ones(
            projected_image_embeds.shape[:2], 
            dtype=torch.long, 
            device=projected_image_embeds.device
        )

        combined_embeds = torch.cat([text_embeds, projected_image_embeds], dim=1)
        combined_attention_mask = torch.cat([article_attention_mask, image_attention_mask], dim=1)

        # --- 4. Pass to mBART Decoder ---
        outputs = self.mbart(
            encoder_outputs=(combined_embeds,),
            attention_mask=combined_attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate(self, article_input_ids, article_attention_mask, image_pixel_values, **gen_kwargs):
        self.eval()
        
        # --- 1. Get Image Embeddings ---
        image_embeds = self.vit(
            pixel_values=image_pixel_values
        ).last_hidden_state

        # --- 2. Get Text Embeddings ---
        text_embeds = self.mbart.model.encoder(
            input_ids=article_input_ids,
            attention_mask=article_attention_mask
        ).last_hidden_state

        # --- 3. Fuse Embeddings ---
        projected_image_embeds = self.projection_layer(image_embeds)
        image_attention_mask = torch.ones(
            projected_image_embeds.shape[:2], 
            dtype=torch.long, 
            device=projected_image_embeds.device
        )
        combined_embeds = torch.cat([text_embeds, projected_image_embeds], dim=1)
        combined_attention_mask = torch.cat([article_attention_mask, image_attention_mask], dim=1)

        encoder_outputs_object = BaseModelOutput(
            last_hidden_state=combined_embeds
        )

        generated_ids = self.mbart.generate(
            encoder_outputs=encoder_outputs_object,
            attention_mask=combined_attention_mask,
            **gen_kwargs
        )
        
        return generated_ids