from __future__ import annotations

import torch
from torch import Tensor, nn


class SkipGram(nn.Module):
    """Skip-gram model for learning token embeddings.

    Given a center token id, this model:
    1) looks up center-token embedding,
    2) projects embedding to vocabulary logits,
    3) predicts likely context tokens around the center token.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.output = nn.Linear(embedding_dim, vocab_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters with small values for stable training."""
        nn.init.uniform_(self.embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.output.weight, a=-0.1, b=0.1)
        nn.init.zeros_(self.output.bias)

    def _center_embedding(self, center_tokens: Tensor) -> Tensor:
        """Return center-token embedding for each example.

        Args:
            center_tokens: LongTensor of shape [batch_size] or [batch_size, 1].
        Returns:
            Embedding tensor with shape [batch_size, embedding_dim].
        """
        if center_tokens.dim() == 2:
            if center_tokens.size(1) != 1:
                raise ValueError(
                    "center_tokens with rank 2 must have shape [batch_size, 1]"
                )
            center_tokens = center_tokens.squeeze(1)
        elif center_tokens.dim() != 1:
            raise ValueError(
                "center_tokens must have shape [batch_size] or [batch_size, 1]"
            )

        return self.embedding(center_tokens)  # [B, E]

    def forward(self, center_tokens: Tensor) -> Tensor:
        """Return context-token logits.

        Args:
            center_tokens: LongTensor with shape [batch_size] or [batch_size, 1].
        Returns:
            Logits tensor with shape [batch_size, vocab_size].
        """
        center_embed = self._center_embedding(center_tokens)
        return self.output(center_embed)

    def probabilities(self, center_tokens: Tensor) -> Tensor:
        """Return context-token probabilities with softmax."""
        return torch.softmax(self.forward(center_tokens), dim=-1)

    def log_probabilities(self, center_tokens: Tensor) -> Tensor:
        """Return context-token log-probabilities."""
        return torch.log_softmax(self.forward(center_tokens), dim=-1)

    def embedding_weight(self) -> Tensor:
        """Return learned token embedding matrix [vocab_size, embedding_dim]."""
        return self.embedding.weight
