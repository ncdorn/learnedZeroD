This package takes a 0D cardiovascular flow model as input and replaces the resistances and inductances for vessels and junctions with values predicted by pretrained neural networks from the vessel or junction geometry.  See [Rubio et. al.](https://arxiv.org/abs/2604.01549) for more details.  The 0D input files considered here are those compatible with [svZeroDSolver](https://github.com/SimVascular/svZeroDSolver), a forward solver for 0D cardiovascular lumped parameter networks.

Developed by Natalia Rubio as part of Ph.D. at Stanford University.  Subject to Stanford IP policy.  Please do not delete or move this repository without notifying me (Natalia).

### Installation

The following command installs this package and its dependencies:

```bash
conda create -n learnedzerod python=3.11 -y && \
conda activate learnedzerod && \
pip install -e ".[dev]"
```

### Execution

To run learned-zerod, run the following command, populating the inputs for your purposes.

```bash
learned-zerod\
  --anatomy aortic \ # Type of anatomies the applied neural networks were trained on.  Choose from "aortic", "aortofemoral", "pulmonary" or "all"
  --zerod-json /path/case.json \ # Path to your original 0D input file (generate with SimVascular)
  --centerline-vtp /path/unsteady_soln.vtp \ # Path to geometry centerline  (generate with SimVascular)
  --svzerod /path/to/svZeroDSolver \  # Path to your installation of svZeroDSolver
  --output-dir /path/to/out # Path to output directory 
```

##### Inputs:

* `anatomy` We provide four sets of pre-trained neural networks - one trained on aortic anatomies, one on aortofemoral anatomies, one on pulmonary anatomies, and one on a dataset containing all anatomies.  Users must choose which set to use.

* `zerod-json` This is the path to your original 0D input file, which will be modified to contain learned resistances and inductances.  This file may be generated from a 3D geometry using the [SimVascular ROM Simulation tool](https://simvascular.github.io/documentation/rom_simulation.html#tool).

* `centerline-vtp` This is a path to the centerline files for your anatomy, which provide a 1D description of the geometry.  The centerline file may be generated with the [SimVascular ROM Simulation tool](https://simvascular.github.io/documentation/rom_simulation.html#tool).

* `svzerod` This is the path to your installation of [svZeroDSolver](https://simvascular.github.io/documentation/rom_simulation.html#0d-solver-install).  `learned-zerod` needs this to run a forward simulation on an intermediate input file to generate estimates of flow splits, which are in turn used as inputs to the neural network.

* `output-dir` This is where `learned-zerod` will save the modified 0D input file.

##### Outputs:

* A 0D input file with learned resistances and inductances saved to `output-dir\[input-file-name]_learned.json`.
