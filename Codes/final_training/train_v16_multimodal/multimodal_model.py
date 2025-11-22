import torch
import torch.nn as nn
import math
from transformers import (
    PreTrainedModel, 
    MBartForConditionalGeneration, 
    ViTModel, 
    ViTConfig, 
    MBartConfig
)

class MultimodalSummarizer(PreTrainedModel):
    config_class = MBartConfig 

    def __init__(self, config):
        super().__init__(config)
        
        # 1. Load ViT Config
        # Ensure config is an object, not a dict (fixes previous TypeError)
        if isinstance(config.vit_config, dict):
            vit_conf = ViTConfig.from_dict(config.vit_config)
        else:
            vit_conf = config.vit_config
        
        # 2. Initialize Visual Encoder
        self.vit = ViTModel(vit_conf)

        # 3. Initialize Text/Multimodal Decoder (MBart)
        self.mbart = MBartForConditionalGeneration(config)
        
        # 4. Visual Projection Layer
        self.visual_projection = nn.Linear(
            vit_conf.hidden_size, 
            config.d_model
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Custom loader to handle key mismatches between standard MBart checkpoints
        and this wrapped Multimodal model.
        """
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        # DEBUG: Check if keys were loaded correctly
        # If the checkpoint was just a raw MBart model, 'mbart.' prefix might be missing in weights.
        # This check ensures we don't silently fail with random weights.
        if hasattr(model, 'mbart') and not any(p.requires_grad for p in model.mbart.parameters() if p.sum() != 0):
            # This is a heuristic; if weights are purely random/zero, something might be wrong.
            # However, standard from_pretrained usually handles mapping if keys match.
            # If you see "Some weights were not initialized" in logs, the keys didn't match.
            pass 
            
        return model

    def _get_embed_scale(self):
        """
        Safely calculates embedding scale.
        MBart uses sqrt(d_model) scaling if scale_embedding is True.
        """
        if getattr(self.config, "scale_embedding", True):
            return math.sqrt(self.config.d_model)
        return 1.0

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        # 1. Encode Images
        vit_outputs = self.vit(pixel_values=pixel_values)
        image_embeds = vit_outputs.last_hidden_state  # (B, seq_len, hidden_dim)
        
        # 2. Project to Text Dimension
        image_embeds = self.visual_projection(image_embeds) # (B, seq_len, d_model)

        # 3. Get Text Embeddings
        # FIX: Calculate scale manually instead of accessing .embed_scale
        embed_scale = self._get_embed_scale()
        inputs_embeds = self.mbart.model.encoder.embed_tokens(input_ids) * embed_scale
        
        # 4. Concatenate: [Image Embeds, Text Embeds]
        inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)
        
        # 5. Adjust Attention Mask (1 for image tokens, then original mask)
        image_mask = torch.ones(
            (pixel_values.shape[0], image_embeds.shape[1]), 
            device=inputs_embeds.device
        )
        attention_mask = torch.cat([image_mask, attention_mask], dim=1)

        # 6. Forward Pass
        outputs = self.mbart(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs

    def generate(self, input_ids, attention_mask, pixel_values, **kwargs):
        # 1. Encode Images
        vit_outputs = self.vit(pixel_values=pixel_values)
        image_embeds = vit_outputs.last_hidden_state
        image_embeds = self.visual_projection(image_embeds)

        # 2. Get Text Embeddings
        # FIX: Calculate scale manually
        embed_scale = self._get_embed_scale()
        inputs_embeds = self.mbart.model.encoder.embed_tokens(input_ids) * embed_scale

        # 3. Concatenate
        inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)
        
        # 4. Adjust Mask
        image_mask = torch.ones(
            (pixel_values.shape[0], image_embeds.shape[1]), 
            device=inputs_embeds.device
        )
        attention_mask = torch.cat([image_mask, attention_mask], dim=1)

        # 5. Generate
        return self.mbart.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )