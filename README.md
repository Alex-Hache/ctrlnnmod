# ctrlnmod
This libray intends to provide tools and implementations for control-oriented neural state-space models.
Its main features include several models based on Linear Matrix Inequalities (LMI) constraints.
Documentation : [![Documentation Status](https://img.shields.io/badge/docs-online-blue.svg)](https://alex-hache.github.io/ctrlnnmod/)


## Requirements
* PyTorch v1.9 >=
* [MOSEK solver](https://www.mosek.com/) can be installed using
 > pip install mosek
  A personal license file can be asked freely for academics
* cvxpy v1.4

## Currently implemented LMIs
* Lyapunov alpha stability for continuous and discrete time LTI models
* $\mathcal{H}^\infty$  norm for continuous and discrete time LTI models see [Real Bounded Lemma](https://en.wikibooks.org/wiki/LMIs_in_Control/KYP_Lemmas/KYP_Lemma_(Bounded_Real_Lemma))
* $\mathcal{H}^2$ norm for both continuous and discrete time LTI models see [System H2 norm](https://en.wikibooks.org/wiki/LMIs_in_Control/pages/LMI_for_System_H2_Norm)
* Lipschitz constant estimation for feedfoward neural networks (with no skip connections) see [Pauli et al. 2021](https://arxiv.org/abs/2005.02929)

## Currently implemented models
From these LMIs one can obtain models with certified properties such as stability, contraction or dissipativity, and incremental boundedness
* Recurrent Equilibrium Networks (REN) implemented in dsicrete-time only for now see [Revay et al. 2023](https://arxiv.org/pdf/2104.05942)
* An incrementally bounded (and stable but not globally contractive) of [GrNSSM](https://arxiv.org/abs/2103.14516) based on real bounded lemma for Taylor Linearizations
* Parameterized Explicit and semi-Implicit linear state-space models using aforementioned LMIs


## Geotorch backend
Finally the parametrizations involving Positive (Semi-) Definite (PSD) matrices are built using the [geotorch](https://github.com/lezcano/geotorch) library.
Since the right inverse methods now involve solving a LMI, cvxpy is now needed, an extended version is then included in this library.


## Experiments framework
In order to train networks on data coming from several experiments and trajectories the class ExperimentsDataset encapsulates different experiments
and is readily compatible with DataLoader. 


This library is the summary of my PhD's work and is still currently under development. 

### Future updates
* Manage compatibility with torchdiffeq integrators
* Implement stability-based initializations strategies
* And enabling gradient and hessian logging with TensorBoardX and [PyHessian](https://github.com/amirgholami/PyHessian)

Feel free to contribute and you can contact me at alexandre DOT hache AT imt-atlantique DOT fr
