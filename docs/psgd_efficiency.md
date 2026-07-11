# PSGD Efficiency

PSGD exposes two independent memory/compute tradeoffs.

## `store_triu_as_line`

`store_triu_as_line=True` stores each triangular preconditioner factor as its upper triangle, halving factor storage.
Reconstructing the matrix costs memory bandwidth; a high-overhead benchmark measured up to 58% longer optimizer steps.
Larger parameter tensors and batches amortize that cost.

![PSGD line-storage benchmark](assets/psgd_efficiency_triu_as_line.png)

## Cached preconditioner

`PSGDKron(cached=True)` stores the precomputed preconditioning matrix instead of rebuilding it from the factors on each
step. This reduces step compute and roughly doubles preconditioner storage. The default is `cached=False`.

![PSGD cache benchmark](assets/psgd_efficiency_cache.png)

Combining `cached=True` with `store_triu_as_line=True` reduces the combined factor-and-cache cost from roughly twice the
factor size to one and a half times it.

![PSGD cached line-storage benchmark](assets/psgd_efficiency_cache_triu_as_line.png)
