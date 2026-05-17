from .base import NegativeSampler
from .random import RandomNegativeSampler


def build_sampler(cfg, num_books: int):
    return RandomNegativeSampler(num_books=num_books, seed=cfg.seed)
