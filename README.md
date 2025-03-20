# Brute-Force Numerical Dust Size Evolution

Numerical "brute-force" implementation of a Smoluchowski-like equation for coagulation and fragmentation of dust grains in planet-forming disks.

### Jan 2024 : Start Simple

Implement simplified Podolak algorithm (Brauer et al. 2008; Appendix A.1). Treats coagulation only (no fragmentation, so single kernel), conserving total mass and particle number density with discretization.

### Feb 2024 : Modified Podolak Algorithm

Implement modified Podolak algorithm (Brauer et al. 2008; Appendix A.2), which analytically reformulates certain components of the simpler numerical scheme to account for numerical precision and error due to the large ranges in orders of magnitude involved.
