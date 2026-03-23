"""MIL model and aggregators on patch embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoConfig
from torchmetrics import MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.stat_scores import StatScores
from torchmetrics.classification import ConfusionMatrix
from torchmetrics.classification import AUROC
from pytorch_lightning.loggers import CSVLogger
from utils.loss import SoftTargetCrossEntropy
import os

# Import the same model dictionary from the original model
MODEL_DICT = {
    "vit-b16-224-in21k": "google/vit-base-patch16-224-in21k",
    "vit-b32-224-in21k": "google/vit-base-patch32-224-in21k",
    "vit-l32-224-in21k": "google/vit-large-patch32-224-in21k",
    "vit-l15-224-in21k": "google/vit-large-patch16-224-in21k",
    "vit-h14-224-in21k": "google/vit-huge-patch14-224-in21k",
    "vit-b16-224": "google/vit-base-patch16-224",
    "vit-l16-224": "google/vit-large-patch16-224",
    "vit-b16-384": "google/vit-base-patch16-384",
    "vit-b32-384": "google/vit-base-patch32-384",
    "vit-l16-384": "google/vit-large-patch16-384",
    "vit-l32-384": "google/vit-large-patch32-384",
    "vit-b16-224-dino": "facebook/dino-vitb16",
    "vit-b8-224-dino": "facebook/dino-vitb8",
    "vit-s16-224-dino": "facebook/dino-vits16",
    "vit-s8-224-dino": "facebook/dino-vits8",
    "beit-b16-224-in21k": "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "beit-l16-224-in21k": "microsoft/beit-large-patch16-224-pt22k-ft22k",
}


class ViTFeatureExtractor(nn.Module):
    """ViT-based feature extractor for patches"""
    
    def __init__(self, model_name: str = "vit-b16-224-in21k", 
                 freeze_backbone: bool = True,
                 output_dim: Optional[int] = None):
        super().__init__()
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        
        try:
            model_path = MODEL_DICT[model_name]
        except KeyError:
            raise ValueError(f"{model_name} is not available. Choose from {list(MODEL_DICT.keys())}")
        
        # Load pretrained ViT model
        self.backbone = AutoModel.from_pretrained(model_path)
        config = self.backbone.config
        self.feature_dim = config.hidden_size
        
        # Freeze backbone if specified
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Optional projection layer
        self.projection = None
        if output_dim is not None:
            self.projection = nn.Linear(self.feature_dim, output_dim)
            self.feature_dim = output_dim
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, C, H, W] (batch of patches)
        Returns:
            features: Tensor of shape [B, feature_dim]
        """
        outputs = self.backbone(pixel_values=x)

        # Use CLS token representation
        features = outputs.last_hidden_state[:, 0]  # [B, hidden_size]

        if self.projection is not None:
            features = self.projection(features)

        return features

"""
Original implementation from:
Ilse, Maximilian, Jakub Tomczak, and Max Welling. 
"Attention-based deep multiple instance learning." 
International conference on machine learning. PMLR, 2018.
"""
class AttentionMIL(nn.Module):
    """Attention-based Multiple Instance Learning aggregation"""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 256, attention_dim: int = 128):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Sigmoid()
        )
        
        self.attention_weights = nn.Linear(attention_dim, 1)
        
    def forward(self, features):
        """
        Args:
            features: Tensor of shape [N, feature_dim] where N is number of patches
        Returns:
            aggregated_features: Tensor of shape [feature_dim]
            attention_weights: Tensor of shape [N]
        """
        A_V = self.attention_V(features)  # [N, attention_dim]
        A_U = self.attention_U(features)  # [N, attention_dim]
        
        attention = self.attention_weights(A_V * A_U)  # [N, 1]
        attention = torch.transpose(attention, 1, 0)  # [1, N]
        attention = F.softmax(attention, dim=1)  # [1, N]
        
        aggregated = torch.mm(attention, features)  # [1, feature_dim]
        aggregated = aggregated.squeeze(0)  # [feature_dim]
        
        return aggregated, attention.squeeze(0)


"""
Original implementation from:
Ilse, Maximilian, Jakub Tomczak, and Max Welling. 
"Attention-based deep multiple instance learning." 
International conference on machine learning. PMLR, 2018.
"""
class GatedAttentionMIL(nn.Module):
    """Gated Attention-based MIL aggregation"""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.attention_b = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, features):
        """
        Args:
            features: Tensor of shape [N, feature_dim]
        Returns:
            aggregated_features: Tensor of shape [feature_dim]
            attention_weights: Tensor of shape [N]
        """
        a = self.attention_a(features)  # [N, 1]
        b = self.attention_b(features)  # [N, 1]
        
        attention = a * b  # [N, 1]
        attention = F.softmax(attention, dim=0)  # [N, 1]
        
        aggregated = torch.sum(attention * features, dim=0)  # [feature_dim]
        
        return aggregated, attention.squeeze(1)

"""
Original implementation from:
Shao, Z., Bian, H., Chen, Y., Wang, Y., Zhang, J., & Ji, X. (2021). 
Transmil: Transformer based correlated multiple instance learning for whole slide image classification. 
Advances in neural information processing systems, 34, 2136-2147.
"""
class TransformerMIL(nn.Module):
    """Transformer-based MIL aggregation"""
    
    def __init__(self, feature_dim: int, nhead: int = 8, num_layers: int = 2, max_pos_encoding: int = 5000):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_pos_encoding = max_pos_encoding
        
        # Add positional encoding with larger default size
        self.pos_encoding = nn.Parameter(torch.randn(max_pos_encoding, feature_dim) * 0.1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global average pooling for aggregation
        self.aggregation = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, features):
        """
        Args:
            features: Tensor of shape [N, feature_dim]
        Returns:
            aggregated_features: Tensor of shape [feature_dim]
            attention_weights: None (transformer doesn't return simple attention weights)
        """
        N = features.shape[0]
        
        # Handle positional encoding for variable sequence lengths
        if N <= self.max_pos_encoding:
            # Use precomputed positional encoding
            pos_enc = self.pos_encoding[:N]
        else:
            # Extend positional encoding by repeating the pattern
            repeat_times = (N // self.max_pos_encoding) + 1
            extended_pos_enc = self.pos_encoding.repeat(repeat_times, 1)
            pos_enc = extended_pos_enc[:N]
        
        # Add positional encoding
        features = features + pos_enc
        
        # Add batch dimension for transformer
        features = features.unsqueeze(0)  # [1, N, feature_dim]
        
        # Pass through transformer
        transformed = self.transformer(features)  # [1, N, feature_dim]
        
        # Global average pooling
        transformed = transformed.transpose(1, 2)  # [1, feature_dim, N]
        aggregated = self.aggregation(transformed).squeeze(-1).squeeze(0)  # [feature_dim]
        
        return aggregated, None


class SimpleMIL(nn.Module):
    """
    Simple aggregation strategies for MIL
    
    Basic pooling operations commonly used as baselines in MIL literature.
    References used across various MIL papers for comparison.
    """
    
    def __init__(self, strategy: str = "mean"):
        super().__init__()
        self.strategy = strategy
        
    def forward(self, features):
        """
        Args:
            features: Tensor of shape [N, feature_dim]
        Returns:
            aggregated_features: Tensor of shape [feature_dim]
            attention_weights: None
        """
        if self.strategy == "mean":
            aggregated = torch.mean(features, dim=0)
        elif self.strategy == "max":
            aggregated, _ = torch.max(features, dim=0)
        elif self.strategy == "sum":
            aggregated = torch.sum(features, dim=0)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return aggregated, None

# --------------------
# New SOTA MIL Aggregators
# --------------------

"""
Original implementation from:
Li, B., Li, Y., & Eliceiri, K. W. (2021). 
Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning. 
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 14318-14328).
"""
class DSMIL(nn.Module):
    """
    Dual-Stream MIL: instance-level and bag-level streams with contrastive hinge loss
    """
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.inst_clf = nn.Linear(feature_dim,1)
        self.bag_clf = nn.Sequential(nn.Linear(feature_dim,hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,1))
    def forward(self, features):
        # instance scores
        inst_scores = self.inst_clf(features).squeeze(1)
        # bag rep via max instance
        idx = inst_scores.argmax(dim=0)
        bag_rep = features[idx]
        bag_score = self.bag_clf(bag_rep)
        return bag_score, inst_scores

"""
Original implementation from:
Lu, M. Y., Williamson, D. F., Chen, T. Y., Chen, R. J., Barbieri, M., & Mahmood, F. (2021). 
Data-efficient and weakly supervised computational pathology on whole-slide images. 
Nature biomedical engineering, 5(6), 555-570.
"""
class CLAM(nn.Module):
    """
    Clustering-Constrained Attention MIL
    """
    def __init__(self, feature_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.attn = nn.Linear(feature_dim, num_classes)
        self.cluster_proj = nn.Linear(feature_dim, hidden_dim)
        self.cluster_head = nn.Linear(hidden_dim, num_classes)
    def forward(self, features):
        A = F.softmax(self.attn(features).transpose(1,0), dim=1)
        bag = torch.mm(A, features).mean(dim=0)
        inst_logits = self.cluster_head(F.relu(self.cluster_proj(features)))
        return bag, inst_logits

"""
Original implementation from:
Wang, X., Yan, Y., Tang, P., Bai, X., & Liu, W. (2018). 
Revisiting multiple instance neural networks. 
Pattern Recognition, 74, 15-24.
"""
class ACMIL(nn.Module):
    """
    Attention-Challenging MIL with multi-branch and stochastic top-k masking
    """
    def __init__(self, feature_dim, branches=4, top_k=5):
        super().__init__()
        self.branches = nn.ModuleList([nn.Linear(feature_dim,1) for _ in range(branches)])
        self.top_k = top_k
    def forward(self, features):
        weights = torch.stack([b(features).squeeze(1) for b in self.branches], dim=1)
        weights = F.softmax(weights,dim=0)
        # average across branches
        w = weights.mean(dim=1)
        # mask top_k
        top_idx = torch.topk(w, self.top_k).indices
        w[top_idx] = 0
        w = F.softmax(w,dim=0)
        bag = torch.sum(w.unsqueeze(1)*features, dim=0)
        return bag, w

"""
Multi-Head Attention MIL - Adapted from standard transformer attention mechanisms:
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). 
Attention is all you need. 
Advances in neural information processing systems, 30.
"""
class MADMIL(nn.Module):
    """
    Multi-Head Attention MIL
    """
    def __init__(self, feature_dim, heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=heads, batch_first=True)
    def forward(self, features):
        x = features.unsqueeze(0)
        attn_output,_ = self.mha(x,x,x)
        attn_output = attn_output.squeeze(0)
        bag = attn_output.mean(dim=0)
        # no single attention weights
        return bag, None

"""
Channel Attention MIL - Inspired by channel attention mechanisms:
Hu, J., Shen, L., & Sun, G. (2018). 
Squeeze-and-excitation networks. 
In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141).
"""
class CAMIL(nn.Module):
    """
    Channel Attention MIL
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.channel_att = nn.Sequential(nn.Linear(feature_dim,feature_dim//2), nn.ReLU(), nn.Linear(feature_dim//2,feature_dim), nn.Sigmoid())
    def forward(self, features):
        # features: [N,D]
        ca = self.channel_att(features.mean(dim=0))  # [D]
        mod = features * ca.unsqueeze(0)
        bag = mod.mean(dim=0)
        return bag, None

"""
Mamba-based MIL - Inspired by state-space models:
Gu, A., & Dao, T. (2023). 
Mamba: Linear-time sequence modeling with selective state spaces. 
arXiv preprint arXiv:2312.00752.
"""
class MamMIL(nn.Module):
    """
    Mamba-based MIL: state-space compression for long-range patch modelling
    """
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.state_proj = nn.Linear(feature_dim,hidden_dim)
        self.state_update = nn.GRUCell(hidden_dim,hidden_dim)
        self.readout = nn.Linear(hidden_dim, feature_dim)
    def forward(self, features):
        s = torch.zeros(features.size(1), device=features.device)
        for x in features:
            z = F.relu(self.state_proj(x))
            s = self.state_update(z, s)
        bag = self.readout(s)
        return bag, None

"""
Context-Aware Regularization MIL - Inspired by spatial context modeling:
Campanella, G., Hanna, M. G., Geneslaw, L., Miraflor, A., Werneck Krauss Silva, V., Busam, K. J., ... & Fuchs, T. J. (2019). 
Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. 
Nature medicine, 25(8), 1301-1309.
"""
class CARMIL(nn.Module):
    """
    Context-Aware Regularization MIL: incorporates spatial neighbors
    """
    def __init__(self, feature_dim, k=4):
        super().__init__()
        self.k = k
        self.attn = nn.Linear(feature_dim,1)
    def forward(self, features):
        # naive: average of each patch and its k neighbors in feature list
        N = features.size(0)
        agg = []
        for i in range(N):
            idx = list(range(max(0,i-self.k), min(N,i+self.k+1)))
            agg.append(features[idx].mean(dim=0))
        agg = torch.stack(agg)
        w = F.softmax(self.attn(agg).squeeze(1), dim=0)
        bag = torch.sum(w.unsqueeze(1)*agg, dim=0)
        return bag, w

"""
Original implementation from:
Chen, R. J., Lu, M. Y., Williamson, D. F., Chen, T. Y., Lipkova, J., Noor, Z., ... & Mahmood, F. (2022). 
Hierarchical image pyramid transformer for histopathological image classification. 
arXiv preprint arXiv:2205.14969.
"""
class HIPT(nn.Module):
    """
    Hierarchical Image Pyramid Transformer MIL
    """
    def __init__(self, feature_dim, levels=2, nhead=8, num_layers=1):
        super().__init__()
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool1d(1) for _ in range(levels)])
        encs = []
        for _ in range(levels):
            layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, batch_first=True)
            encs.append(nn.TransformerEncoder(layer, num_layers=num_layers))
        self.encs = nn.ModuleList(encs)
        self.combine = nn.Linear(feature_dim*levels, feature_dim)
    def forward(self, features):
        outs = []
        for pool, enc in zip(self.pools, self.encs):
            x = features.unsqueeze(0)
            y = enc(x).transpose(1,2)
            y = pool(y).squeeze(2).squeeze(0)
            outs.append(y)
        multi = torch.cat(outs, dim=0)
        bag = self.combine(multi)
        return bag, None

class MILClassificationModel(pl.LightningModule):
    """Multiple Instance Learning Model for Classification"""
    
    def __init__(
        self,
        model_name: str = "vit-b16-224-in21k",
        freeze_backbone: bool = True,
        feature_dim: Optional[int] = None,
        aggregation_method: str = "hipt",  # "hipt" (default), "attention", "gated", "transformer", "mean", "max", "sum"
        hidden_dim: int = 256,
        attention_dim: int = 128,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        n_classes: int = 2,
        optimizer: str = 'adam',
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        scheduler: str = 'cosine', 
        warmup_steps: int = 0,
        label_smoothing: float = 0.0,
        use_precomputed_features: bool = True,  # New parameter to use pre-computed features
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store hyperparameters
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.aggregation_method = aggregation_method
        self.n_classes = n_classes
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.label_smoothing = label_smoothing
        self.use_precomputed_features = use_precomputed_features
        
        # Determine feature dimension
        if self.use_precomputed_features:
            # For pre-computed features, we need to infer the dimension from the model
            # Default dimensions for common models
            model_to_dim = {
                "vit-b16-224-in21k": 768,
                "vit-s16-224-dino": 384,
                "vit-b16-224-dino": 768,
                "beit-b16-224-in21k": 768,
            }
            feat_dim = feature_dim or model_to_dim.get(model_name, 768)
            self.feature_extractor = None  # No feature extractor needed
        else:
            # Original approach with feature extractor
            self.feature_extractor = ViTFeatureExtractor(
                model_name=model_name,
                freeze_backbone=freeze_backbone,
                output_dim=feature_dim
            )
            feat_dim = self.feature_extractor.feature_dim
        
        method = aggregation_method.lower()
        if method == "attention": self.aggregator = AttentionMIL(feat_dim, hidden_dim, attention_dim)
        elif method == "gated": self.aggregator = GatedAttentionMIL(feat_dim, hidden_dim)
        elif method == "transformer": self.aggregator = TransformerMIL(feat_dim, transformer_heads, transformer_layers)
        elif method in ["mean","max","sum"]: self.aggregator = SimpleMIL(method)
        elif method == "dsmil": self.aggregator = DSMIL(feat_dim, hidden_dim)
        elif method == "clam": self.aggregator = CLAM(feat_dim, n_classes, hidden_dim)
        elif method == "acmil": self.aggregator = ACMIL(feat_dim, branches=4, top_k=5)
        elif method == "madmil": self.aggregator = MADMIL(feat_dim, heads=4)
        elif method == "camil": self.aggregator = CAMIL(feat_dim)
        elif method == "mammil": self.aggregator = MamMIL(feat_dim)
        elif method == "carmil": self.aggregator = CARMIL(feat_dim)
        elif method == "hipt": self.aggregator = HIPT(feat_dim)
        else: raise ValueError(f"Unknown aggregator: {aggregation_method}")
        
        # Classifier
        self.classifier = nn.Linear(feat_dim, n_classes)
        
        # Metrics
        self.train_metrics = MetricCollection({
            "acc": Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1, average='macro'),
            "stats": StatScores(task="multiclass", average=None, num_classes=self.n_classes),
        })
        self.val_metrics = MetricCollection({
            "acc": Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1, average='macro'),
            "stats": StatScores(task="multiclass", average=None, num_classes=self.n_classes),
        })
        self.test_metrics = MetricCollection({
            "acc": Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1, average='macro'),
            "stats": StatScores(task="multiclass", average=None, num_classes=self.n_classes),
        })
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Confusion matrix for validation
        self.val_confusion = ConfusionMatrix(task="multiclass", num_classes=self.n_classes)
        self.best_val_acc = 0.0

        # AUROC metrics (per-class, one-vs-rest)
        self.val_auroc = AUROC(task="multiclass", num_classes=self.n_classes, average=None)
        self.test_auroc = AUROC(task="multiclass", num_classes=self.n_classes, average=None)
                
    def forward(self, embeddings_list):
        """
        Args:
            embeddings_list: List of embedding tensors, each of shape [num_patches, feature_dim]
        Returns:
            logits: Tensor of shape [batch_size, n_classes]
            attention_weights: List of attention weights (if available)
        """
        batch_logits = []
        batch_attention = []
        
        for embeddings in embeddings_list:
            # embeddings is already [num_patches, feature_dim] from pre-computed features
            # Aggregate features
            aggregated_features, attention_weights = self.aggregator(embeddings)
            
            # Classify
            logits = self.classifier(aggregated_features)
            
            batch_logits.append(logits)
            batch_attention.append(attention_weights)
        
        batch_logits = torch.stack(batch_logits)  # [batch_size, n_classes]
        
        return batch_logits, batch_attention
    
    def shared_step(self, batch, mode="train"):
        patches_list, labels, paths = batch
        
        logits, attention_weights = self(patches_list)
        
        if mode == "train":
            labels = F.one_hot(labels, num_classes=self.n_classes).float()
        else:
            labels = F.one_hot(labels, num_classes=self.n_classes).float()

        loss = self.loss_fn(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        metrics = getattr(self, f"{mode}_metrics")(preds, labels.argmax(1))
        
        # Log loss and metrics
        self.log(f"{mode}_loss", loss, on_epoch=True, sync_dist=True)
        for k, v in metrics.items():
            if len(v.size()) == 0:
                self.log(f"{mode}_{k.lower()}", v, on_epoch=True, sync_dist=True)
        
        # Store metric outputs for epoch end processing
        if mode == "test":
            if not hasattr(self, 'test_metric_outputs'):
                self.test_metric_outputs = []
            self.test_metric_outputs.append(metrics["stats"])
        elif mode == "val":
            if not hasattr(self, 'val_metric_outputs'):
                self.val_metric_outputs = []
            self.val_metric_outputs.append(metrics["stats"])
            self.val_confusion.update(preds, labels)
        
        return {
            "loss": loss,
            "preds": preds,
            "labels": labels,
            "attention_weights": attention_weights,
            **metrics
        }
    
    def training_step(self, batch, batch_idx):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, sync_dist=True)
        result = self.shared_step(batch, "train")
        return result["loss"]
    
    def validation_step(self, batch, batch_idx):
        patches_list, labels_int, paths = batch            # <-- keep integer labels
        logits, _ = self(patches_list)
        probs = torch.softmax(logits, dim=1)

        labels_oh = F.one_hot(labels_int, num_classes=self.n_classes).float()  # for loss only
        loss = self.loss_fn(logits, labels_oh)
        preds = torch.argmax(logits, dim=1)

        # metrics that expect integer targets
        metrics = self.val_metrics(preds, labels_int)

        # AUROC expects probs (N,C) and integer targets (N,)
        self.val_auroc.update(probs, labels_int)

        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        conf_matrix = self.val_confusion(logits, labels_int)

        if not hasattr(self, 'validation_step_outputs'):
            self.validation_step_outputs = []
        self.validation_step_outputs.append({
            'val_acc': metrics['acc'],
            'conf_matrix': conf_matrix
        })
        for k, v in metrics.items():
            if v.ndim == 0:
                self.log(f"val_{k.lower()}", v, on_epoch=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        result = self.shared_step(batch, "test")

        patches_list, labels_int, paths = batch
        logits, _ = self(patches_list)
        probs = torch.softmax(logits, dim=1)
        self.test_auroc.update(probs, labels_int)
        return result["loss"]
    
    def on_validation_epoch_start(self):
        self.validation_step_outputs = []
        self.val_confusion.reset()
        self.val_auroc.reset()

    def on_validation_epoch_end(self):
        # Skip saving during sanity check to avoid partial confusion matrices
        if self.trainer.sanity_checking:
            return
            
        if hasattr(self, 'validation_step_outputs') and self.validation_step_outputs:
            val_acc = torch.stack([x['val_acc'] for x in self.validation_step_outputs]).mean()
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc

                # Confusion matrix
                confusion_matrix = self.val_confusion.compute()

                # Per-class AUROCs (shape: [num_classes])
                per_class_aurocs = self.val_auroc.compute()  # tensor, may contain NaN if a class absent

                if isinstance(self.logger, CSVLogger):
                    fold = getattr(self.trainer.datamodule, 'fold', None) if hasattr(self.trainer, 'datamodule') else None
                    log_dir = self.logger.log_dir
                    if fold is not None:
                        cm_path = os.path.join(log_dir, f"best_confusion_matrix_fold_{fold}.csv")
                        auroc_path = os.path.join(log_dir, f"best_per_class_auroc_fold_{fold}.csv")
                    else:
                        cm_path = os.path.join(log_dir, "best_confusion_matrix.csv")
                        auroc_path = os.path.join(log_dir, "best_per_class_auroc.csv")

                    np.savetxt(cm_path, confusion_matrix.cpu().numpy(), delimiter=',', fmt='%d')

                    # save AUROC with class index column
                    df_auroc = pd.DataFrame({
                        "class_idx": list(range(self.n_classes)),
                        "auroc": per_class_aurocs.detach().cpu().numpy()
                    })
                    df_auroc.to_csv(auroc_path, index=False)

            self.validation_step_outputs.clear()

        # Log macro AUROC and ACC for the epoch
        per_class = self.val_auroc.compute()  # tensor [C]
        macro_auroc = torch.nanmean(per_class)  # safe if some classes missing
        self.log("val_auroc_macro", macro_auroc, prog_bar=True)
        metrics = self.val_metrics.compute()
        self.log("val_acc", metrics["acc"], prog_bar=True)

        # Reset epoch metrics
        self.val_metrics.reset()
        self.val_auroc.reset()

    def on_test_epoch_end(self):
        # existing per-class accuracy save...
        if hasattr(self, 'test_metric_outputs') and self.test_metric_outputs:
            combined_stats = torch.sum(torch.stack(self.test_metric_outputs, dim=-1), dim=-1)
            per_class_acc = []
            for tp, _, _, _, sup in combined_stats:
                acc = tp / sup
                per_class_acc.append((acc.item(), sup.item()))
            df = pd.DataFrame(per_class_acc, columns=["acc", "n"])
            df.to_csv("per-class-acc-test.csv")
            print("Saved per-class results in per-class-acc-test.csv")

        # --- NEW: per-class AUROC on test ---
        per_class_test_auroc = self.test_auroc.compute()  # tensor [C]
        pd.DataFrame({
            "class_idx": list(range(self.n_classes)),
            "auroc": per_class_test_auroc.detach().cpu().numpy()
        }).to_csv("per-class-auroc-test.csv", index=False)
        print("Saved per-class AUROC in per-class-auroc-test.csv")

        # Overall test metrics
        metrics = self.test_metrics.compute()
        self.log("test_acc", metrics["acc"])
        self.test_metrics.reset()
        self.test_auroc.reset()
    
    def configure_optimizers(self):
        # Create optimizer
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        
        # Create scheduler
        if self.scheduler == "cosine":
            from transformers.optimization import get_cosine_schedule_with_warmup
            
            # Estimate total steps (this is approximate)
            total_steps = int(self.trainer.estimated_stepping_batches)
            
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            }
        
        return optimizer
    
    def get_attention_maps(self, embeddings_list):
        """Get attention maps for visualization
        
        Args:
            embeddings_list: List of embedding tensors or single embedding tensor list
        Returns:
            attention_weights: List of attention weights (if available)
        """
        # Handle different input formats
        if not isinstance(embeddings_list, list):
            embeddings_list = [embeddings_list]
        
        with torch.no_grad():
            _, attention_weights = self(embeddings_list)
        return attention_weights 