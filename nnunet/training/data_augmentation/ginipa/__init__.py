"""
nunetlite.training.augmentations.ginipa

Paper https://arxiv.org/pdf/2111.12525

Workflow of the causality-inspired GIN-IPA:
    1. GIN: random shallow network g1, g2 
    2. IPA: random biasfield that combine two GIN outputs
        - t1 = b*g1+(1-b)*g2, t2 = (1-b)*g1+b*g2
    3. Compute loss: T1, T2, KL
"""
