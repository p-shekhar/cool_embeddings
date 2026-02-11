from __future__ import annotations

import torch
from torch import Tensor, nn


class NeuralProbabilisticLanguageModel(nn.Module):
    """Bengio et al. (2003) neural probabilistic language model.

    Given a fixed-size context of token ids (n-1 previous words), this model:
    1) looks up and concatenates context embeddings,
    2) applies a hidden feed-forward layer with tanh activation,
    3) predicts next-token logits over the vocabulary.
    """

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        embedding_dim: int,
        hidden_dim: int,
        use_direct_connection: bool = True,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if context_size <= 0:
            raise ValueError("context_size must be > 0")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_direct_connection = use_direct_connection

        flattened_dim = context_size * embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.hidden = nn.Linear(flattened_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.direct = (
            nn.Linear(flattened_dim, vocab_size, bias=False)
            if use_direct_connection
            else None
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters with small values for stable training."""
        nn.init.uniform_(self.embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.hidden.weight, a=-0.1, b=0.1)
        nn.init.zeros_(self.hidden.bias)
        nn.init.uniform_(self.output.weight, a=-0.1, b=0.1)
        nn.init.zeros_(self.output.bias)
        if self.direct is not None:
            nn.init.uniform_(self.direct.weight, a=-0.1, b=0.1)

    def _flattened_context(self, context_tokens: Tensor) -> Tensor:
        """Lookup context token embeddings and flatten them.

        Args:
            context_tokens: LongTensor of shape [batch_size, context_size].
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
        return embedded.reshape(embedded.size(0), -1)  # [B, C * E]

    def forward(self, context_tokens: Tensor) -> Tensor:
        """Return next-token logits.

        Args:
            context_tokens: LongTensor with shape [batch_size, context_size].
        Returns:
            Logits tensor with shape [batch_size, vocab_size].
        """
        x = self._flattened_context(context_tokens)
        hidden_state = self.activation(self.hidden(x))
        logits = self.output(hidden_state)
        if self.direct is not None:
            logits = logits + self.direct(x)
        return logits

    def probabilities(self, context_tokens: Tensor) -> Tensor:
        """Return next-token probabilities with softmax."""
        return torch.softmax(self.forward(context_tokens), dim=-1)

    def log_probabilities(self, context_tokens: Tensor) -> Tensor:
        """Return next-token log-probabilities."""
        return torch.log_softmax(self.forward(context_tokens), dim=-1)
