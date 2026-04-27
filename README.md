## Script files

- pca_sims.ipynb: 
    * PCA applied to simulated spectra (range from 6173.10077 &Aring; to 6173.599098 &Aring;, i.e. just one absorption line) with doppler correction to the wavelength.
- pca_sims2.ipynb:
    * PCA applied to simulated spectra. We take out the projection over the velocity vector. To do that, we first represent each spectrum as a small shift in the reference spectrum:
    
        ![alt text](imagen-2.png)

        Using the Doppler shift formula $\Delta v / c = \Delta \lambda / \lambda$, we get:

        ![alt text](imagen-1.png)

        Where $S_i$ is a simulated spectrum and $S_0$ is the reference spectrum (mask).

    

        PCA applied to simulated spectra without doppler correction to the wavelength.