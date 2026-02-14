from __future__ import annotations

import torch
from torch import Tensor, nn


class ContinuousBagOfWords(nn.Module):
    """Continuous Bag-of-Words (CBOW) model for learning token embeddings.

    Given a fixed-size context of token ids, this model:
    1) looks up context embeddings,
    2) averages embeddings across context positions,
    3) predicts target-token logits over the vocabulary.
    """

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        embedding_dim: int,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if context_size <= 0:
            raise ValueError("context_size must be > 0")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")

        self.vocab_size = vocab_size
        self.context_size = context_size
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

    def _context_embedding_mean(self, context_tokens: Tensor) -> Tensor:
        """Return mean context embedding for each example.

        Args:
            context_tokens: LongTensor of shape [batch_size, context_size].
        Returns:
            Mean embedding tensor with shape [batch_size, embedding_dim].
        """
        if context_tokens.dim() != 2:
            raise ValueError(
                "context_tokens must have shape [batch_size, context_size]"
            )
        if context_tokens.size(1) != self.context_size:
            raise ValueError(
                f"Expected context size {self.context_size}, "
                f"got {context_tokens.size(1)}"
            )

        embedded = self.embedding(context_tokens)  # [B, C, E]
        return embedded.mean(dim=1)  # [B, E]

    def forward(self, context_tokens: Tensor) -> Tensor:
        """Return target-token logits.

        Args:
            context_tokens: LongTensor with shape [batch_size, context_size].
        Returns:
            Logits tensor with shape [batch_size, vocab_size].
        """
        context_mean = self._context_embedding_mean(context_tokens)
        return self.output(context_mean)

    def probabilities(self, context_tokens: Tensor) -> Tensor:
        """Return target-token probabilities with softmax."""
        return torch.softmax(self.forward(context_tokens), dim=-1)

    def log_probabilities(self, context_tokens: Tensor) -> Tensor:
        """Return target-token log-probabilities."""
        return torch.log_softmax(self.forward(context_tokens), dim=-1)

    def embedding_weight(self) -> Tensor:
        """Return learned token embedding matrix [vocab_size, embedding_dim]."""
        return self.embedding.weight
