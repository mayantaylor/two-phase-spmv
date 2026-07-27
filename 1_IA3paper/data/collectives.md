Results for 1D reduce and broadcast over rows of PEs. 
Generated in Tungsten + Paint and evaluated in simulator and on the WSE3 in July 2026.

scaleP results: generated with 256 32 bit elements

scaleB results: generated with 512 PEs

### Reduce
The Luczynski et al. paper uses the following model:

$$  T_{chain} = B + (2 T_r + 2) * (P-1) $$

With the following assumptions:
- 850 MHz clock rate
- $T_r$ is approximated as 2, based on simulator results. They also say that an earlier work cited $T_r = 7$
- B is the vector length (32-bit elements), or the number of wavelets
- P is the number of PEs. In scaling B, they use P=512.

Our results fit a model with these changes:
- $T_r$ = 5
- We use an extra PE on each end, so substitute $P+1$
- Add additional latency for reticle crossings: 8 reticles in a row of 512 PEs (reticle every 64 PEs), each reticle crossing adds 9 cycle latency. 

$$  T_{chain-updated} = B + (2*5 + 2) * (P+1) + \frac{9P}{64} $$

## Broadcast

The Luczynski et al. paper uses the following model:

$$  T_{bcast} = B + P + 2T_r $$

With the following assumptions:
- 850 MHz clock rate
- $T_r = 2$
- B is the vector length (32-bit elements), or the number of wavelets
- P is the number of PEs.

Our results fit a model with these changes:
- $T_r$ = 5
- There should be a 2 term in front of P. The PE crossing latency is 2 cycles. The reduction model does account for this.
- Add additional latency for reticle crossings: 8 reticles in a row of 512 PEs (reticle every 64 PEs), each reticle crossing adds 9 cycle latency. 

$$  T_{bcast-updated} = B + 2 P + 2T_r + \frac{9P}{64} $$

Also note, whether the broadcast reads to a temporary variable or to a array location (presumably in memory) changes the performance dramatically, as indicated by bcast-totemp vs. bcast-tomem

