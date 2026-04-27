## Script files

- pca_sims.ipynb: 
    * PCA applied to simulated spectra (range from 6173.10077 &Aring; to 6173.599098 &Aring;, i.e. just one absorption line) with doppler correction to the wavelength.
- pca_sims2.ipynb:
    * PCA applied to simulated spectra. We take out the projection over the velocity vector. To do that, we first represent each spectrum as a small shift in the reference spectrum:
    
        $$
        S_i = S_0(\lambda + \Delta \lambda) \cong S_0(\lambda) + \frac{dS_0(\lambda)}{d\lambda}\Delta \lambda
        $$

        Using the Doppler shift formula $\Delta v / c = \Delta \lambda / \lambda$, we get:

        $$S_i - S_0(\lambda) \simeq \frac{dS_0(\lambda)}{d\lambda} \frac{\lambda}{c} \Delta v $$

        Where $S_i$ is a simulated spectrum and $S_0$ is the reference spectrum (mask).

    

        PCA applied to simulated spectra without doppler correction to the wavelength.