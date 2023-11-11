Want to learn the Preal distribution with Pg. there are two section: G generator D discriminator, D want to distinguish the Pg and Preal; but G want to maximum the likely hood to Preal, so that the D can not tell the difference (finally Distingishing success ~ 1/2)
 - from math, after fix G, the D can be trained to certain value
 - then, after fix D, the G can be trained to Pr = Pg, so the loss function can converge at the last
 - by JS divergence as loss function, instability: when Preal and Pg no overlap, the gradient will be 0
 - the solution is WGAN
 - loss function of Earth Mover's distance / Wassersteom distance
 - Gradient Penalty: WGANs often incorporate a gradient penalty (WGAN-GP) to enforce the Lipschitz constraint, which is necessary for the optimal critic (discriminator) to be a 1-Lipschitz function. This technique involves penalizing the norm of the gradient of the critic with respect to its input.