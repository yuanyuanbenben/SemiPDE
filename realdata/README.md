# Real-data applications

Two applications of the framework, each in its own self-contained directory:

- [`invitro_cell/`](invitro_cell/) — PC-3 in-vitro cell culture (scratch assay),
  with linear and porous diffusion variants. Data, code, results and figures are
  included.
- [`vhb/`](vhb/) — multi-rate viscoelasticity of VHB 4905 on a three-dimensional
  finite-element model. Code, calibrated parameters, results and figures are
  included; the external finite-element solver and the experimental data are
  obtained from the authors' public release.

Both applications estimate an unknown nonparametric component jointly with a
small number of physical parameters. The README in each directory gives the
data split, the model, the result tables and the reproduction commands.
