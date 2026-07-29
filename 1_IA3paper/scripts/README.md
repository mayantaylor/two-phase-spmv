### generate-rand-matrices.py

Script to generate test-case matrices. The "spread" parameter controls the density around the diagonal.

Example usage:
- to generate random matrices (no diagonal spread):
``python generate-rand-matrices.py A.npz 65536 65536 random --density .01 --seed 1 ``
- to generate matrices on the "spread" spectrum:
``python generate-rand-matrices.py A.npz 65536 65536 diagonal_spectrum --density .01 --seed 1 --spread .5``
