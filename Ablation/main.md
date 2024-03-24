> Can you evaluate whether the convection/diffusion/forcing/source terms are actually learning those parts of the equation?

**A:** 
Yes. Here, we can use the two open-access datasets (Burgers2d and RD2d) as examples. To obtain the detailed term for convection/diffusion/source terms, we use high-fidelity FDM (finite-difference method) and FVM (finite-volume methods) to compute the corresponding terms. Subsequently, these terms are downsampled to the coarser resolutions aligned with the paper. The $\epsilon$ ($L_2$ relative error) between ground truth and numerical results are $0.0041$ and $0.0032$ in all time steps for Burgers2d and RD2d, respectively. Thus, we can use the results obtained by the numerical methods as reference values to verify this. From the quantitative calculation results (see below) and visual effect (see Appendix), the different terms of PAPM can learn convection/coursing together/source terms with parts of the equation.


| Datasets |  $\epsilon$ | convection $\epsilon$ | diffusion $\epsilon$ | source $\epsilon$ |
|:---:     | :---:   |:---:  |:---:        |:---:  |
| Burgers2d| 0.039   |0.037  | 0.041       | 0.069 | 
| RD2d     | 0.018   | -     | 0.025       | 0.012 |

Polt: 

![](fig/pipline.jpg)

![](fig/pipline.jpg)
