# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred, mask_tokens_out = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for outptu
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]  #bs, num_points, num_out_tokens, h, w
        iou_pred = iou_pred[:, :, mask_slice]  #bs, num_points, num_out_tokens
        mask_tokens_out = mask_tokens_out[:, :, mask_slice] # bs, num_points, num_out_tokens, 256

        # Prepare output
        return masks, iou_pred, mask_tokens_out

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0) # 5, 256
        output_tokens = output_tokens.unsqueeze(0).unsqueeze(0).expand(*sparse_prompt_embeddings.size()[:2], -1, -1)  # bs, num_points, 5, 256
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings.unsqueeze(1), tokens.shape[1], dim=1) # bs, num_points, c, h, w
        src = src + dense_prompt_embeddings  # bs, num_points, c, h, w
        pos_src = torch.repeat_interleave(image_pe.unsqueeze(1), tokens.shape[1], dim=1).repeat(len(src), 1, 1, 1, 1) # bs, num_points, c, h, w


        bs, n, c, h, w = src.shape
        # num_tokens = tokens.size(2)
        src = src.flatten(0, 1)   # bs * num_points, c, h, w               #.reshape(bs * n, c, h, w) 
        pos_src = pos_src.flatten(0, 1)    # bs * num_points, c, h, w      #.reshape(bs * n, c, h, w)
        tokens = tokens.flatten(0, 1) # bs * num_points, 7, c              #.reshape(bs * n, num_tokens, c)  

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)

        
        iou_token_out = hs[:, 0, :]  # bs * num_points, c
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]  # bs * num_points, 4, c

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(bs * n, c, h, w)  # bs * num_points, c, h, w
        upscaled_embedding = self.output_upscaling(src) # bs * num_points, c, h, w
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # bs * num_points, 4, c
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)    # bs * num_points, 4, h * w 

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)  # bs * num_points, 4


        num_tokens = masks.size()[1]
        masks = masks.reshape(bs, n, num_tokens, h, w)  # bs, num_points, 4, h, w 

        c = tokens.size(-1)
        mask_tokens_out = mask_tokens_out.reshape(bs, n, num_tokens, c) # bs, num_points, 4, 256

        c = iou_pred.size()[-1]
        iou_pred = iou_pred.reshape(bs, n, c)   # bs, num_points, c

        

        return masks, iou_pred, mask_tokens_out


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
