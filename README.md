## Script files

- pca_dace.ipynb:
    * PCA applied to data obtained from HARPS-N using dace_query. The script to download the files is available as a raw cell if the files are not in the repository.

- RV_jupiter_7_4.ipynb: backup to pca_dace.ipynb from April 7th 2026.

- pca_sims.ipynb: 
    * PCA applied to a time series of simulated spectra (range from 6173.10077 &Aring; to 6173.599098 &Aring;, i.e. just one absorption line) with doppler correction to the wavelength.

- pca_sims2.ipynb:
    * PCA applied to simulated spectra. We take out the projection over the velocity vector. To do that, we first represent each spectrum as a slightly shifted reference spectrum:
    
        <img src="images/equations/eq1.png" width="300">

        Using the Doppler shift formula $\Delta v / c = \Delta \lambda / \lambda$, we get:

        <img src="images/equations/eq2.png" width="200">

        Where $S_i$ is a simulated spectrum and $S_0$ is the reference spectrum (mask).
        We define $S(t) = S_i - S_0(\lambda)$ and $D = \frac{dS_0(\lambda)}{d\lambda} \frac{\lambda}{c}$. Then, the spectrum without the projection over the velocity vector is:

        <img src="images/equations/eq3.png" width="230">

        Where $<D, S(t)>$ is the inner product. The PCA is applied to a matrix of $S_f$.
   
- pca_sims_raw.ipynb:
    * PCA applied to simulated spectra without doppler correction to the wavelength.

- pca_sims_series.ipynb:
    * PCA applied to multiple time series of spectra, the same way as in pca_sims.ipynb.

- tools.py: functions used in the other scripts.