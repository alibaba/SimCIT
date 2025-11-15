import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from modules.encoder import MLP
from modules.quantization.quantize import VQBottleneck
# from modules.quantization.residual_quantization import RQBottleneck

"""
Spatial Semantic Identifier that fuses multi-source side information via contrastive learning.
"""

class SpatialSemanticIdentifier(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        self.data_sources = model_config['data_sources']
        """encoder config"""
        self.data_input_dims = model_config['encoder'].get('input_dim',  {name: 1024 for name in self.data_sources})
        self.embedding_dim = model_config['encoder'].get('embedding_dim', 128)
        self.encoder_dnn_layers = model_config['encoder'].get('dnn_layers', [512, 384, 256])
        self.encoder_layers = self.build_encoder_layers()
        """attention """
        self.softmax = nn.Softmax(dim=-1).to(self.device)
        self.query = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        ).to(self.device)

        """codebook config"""
        self.n_embed = model_config['codebook'].get('codebook_size', 512)
        self.n_codebooks = model_config['codebook'].get('n_codebooks', 3)
        self.kmeans_init = model_config['codebook'].get('kmeans_init', True)
        self.gumbel_t = model_config['codebook'].get('gumbel_t', 0.2)
        self.sim_vq = model_config['codebook'].get('sim_vq', False)

        self.rq_layers = nn.ModuleList(modules=[
            VQBottleneck(
                embed_dim=self.embedding_dim,
                n_embed=self.n_embed,
                do_kmeans_init=self.kmeans_init,
                sim_vq=self.sim_vq
            ) for i in range(self.n_codebooks)
        ]).to(self.device)

        """contrastive"""
        self.temperature = model_config.get('temperature', 0.1)

        """projection layer"""
        projection_layer_type = model_config['projection'].get('projection_layer', "identity")
        if projection_layer_type == "linear":
            self.projection_layer = nn.Linear(self.embedding_dim, model_config['projection'].get('projection_dim', 128)).to(self.device)
        elif projection_layer_type == "nonlinear":
            self.projection_layer = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),
                                                  nn.ReLU(),
                                                  nn.Linear(self.embedding_dim, model_config['projection'].get('projection_dim', 128))).to(self.device)
        else:
            self.projection_layer = nn.Identity().to(self.device)

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num }
    def build_encoder_layers(self):
        encoder_layers = {}
        for name in self.data_sources:
            if name not in self.data_input_dims:
                raise KeyError
            else:
                setattr(self, f"encoder_{name}", MLP(input_dim=self.data_input_dims[name],
                                           hidden_dims=self.encoder_dnn_layers,
                                           out_dim=self.embedding_dim).to(self.device))
                encoder_layers[name] = getattr(self, f"encoder_{name}")
        return encoder_layers

    def get_semantic_ids(self, x):
        res = x
        codebook_loss = 0
        commitment_loss = 0
        embs, sem_ids = [], [] #, []
        for layer in self.rq_layers:
            # residuals.append(res)
            quantized = layer(res, temperature=self.gumbel_t)
            codebook_loss += quantized.emb_loss
            commitment_loss += quantized.commit_loss
            emb, id = quantized.embeddings, quantized.ids
            res = res - emb
            sem_ids.append(id)
            embs.append(emb)
        embeddings = rearrange(embs, "b h d -> h d b")
        # residuals = rearrange(residuals, "b h d -> h d b")
        sem_ids = rearrange(sem_ids, "b d -> d b")
        return embeddings, sem_ids, codebook_loss.mean(), commitment_loss.mean()


    def batched_contrastive_loss(self, z1, z2, batch_size=4096):
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.temperature)
        indices = torch.arange(0, num_nodes).to(self.device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    @torch.no_grad()
    def get_codes(self, raw_inputs):
        all_embeddings = {}
        # get all the embeddings for different modalities
        for name in self.data_sources:
            feat = raw_inputs[name]
            if name in self.encoder_layers:
                emb = self.encoder_layers[name](feat.to(self.device))
                all_embeddings[name] = emb
            else:
                raise KeyError
        # apply attention to fuse all the embeddings
        attn = torch.cat([self.query(emb) for name, emb in all_embeddings.items()], dim=-1)
        weight = self.softmax(attn)
        fused_embedding = sum([weight[:, i].unsqueeze(dim=1) * emb for i, emb in enumerate(all_embeddings.values())])
        # apply residual quantization
        quantized_encodings, quantized_indices, _, _ = self.get_semantic_ids(fused_embedding)
        return quantized_encodings, quantized_indices
    @property
    def codebooks(self):
        # F.normalize(self._C.data, dim=-1)
        codebooks = torch.stack([F.normalize(layer.embedding.weight, dim=-1) for layer in self.rq_layers], dim=0)
        return codebooks

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def forward(self, raw_inputs):
        all_embeddings = {}
        # get all the embeddings for different modalities
        for name in self.data_sources:
            feat = raw_inputs[name]
            if name in self.encoder_layers:
                emb = self.encoder_layers[name](feat.to(self.device))
                all_embeddings[name] = emb
            else:
                raise KeyError
        # apply attention to fuse all the embeddings
        attn = torch.cat([self.query(emb) for name, emb in all_embeddings.items()], dim=-1)
        weight = self.softmax(attn)
        fused_embedding = sum([weight[:, i].unsqueeze(dim=1) * emb for i, emb in enumerate(all_embeddings.values())])
        # apply residual quantization
        quantized_embedding, _, codebook_loss, commitment_loss = self.get_semantic_ids(fused_embedding)
        # encodings = []
        # for i in range(1, self.n_codebooks+1):
        #     encodings.append( self.projection_layer(quantized_embedding[:, :, 0:i].sum(-1)))
        quantized_embedding = self.projection_layer(quantized_embedding.sum(-1))
        codebook_loss = torch.einsum('mkd,mjd->mkj', self.codebooks, self.codebooks).mean()
        losses = {"codebook_loss": 1. * codebook_loss, "commitment_loss": 0 * commitment_loss}
        for name, emb in all_embeddings.items():
            losses[f"contrastive_{name}_loss"] = self.batched_contrastive_loss(quantized_embedding, self.projection_layer(emb)) # (self.batched_contrastive_loss(quantized_embedding, emb) + self.batched_contrastive_loss(fused_embedding, emb)) / 2
            # losses[f"contrastive_{name}_loss"] = sum([self.batched_contrastive_loss(encoding, self.projection_layer(emb)) for encoding in encodings]) / len(encodings)  # (self.batched_contrastive_loss(quantized_embedding, emb) + self.batched_contrastive_loss(fused_embedding, emb)) / 2
        return losses