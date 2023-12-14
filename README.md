# MISMax:
The purpose of the code in this repository is to identify graphs which (do not) have a permis, as defined in the paper Sequential convergence of Boolean networks and kernels in digraphs, available at https://arxiv.org/abs/2307.05216. 

The script `sans_permis.py` is referenced in the proof of Theorem 4 in the paper (specifically, it is the computer proof for proposition 1, that there exists a permis for: "All graphs of at most seven vertices, except the heptagon $C_7$.")

The script `MISMax.py` was used to efficiently test whether certain graphs of interest were with or without a permis, specifically augmenting odd cycles to obtain graphs which plausibly would not have a permis. That script is not relied on directly for any result proof. 

The script `sans_permis_mais_en_plus_vite.py` is a work in progress to more rapidly exhaustively search for small graphs without a permis by parallelizing/precompiling certain parts of the code. 

