This package takes a 0D cardiovascular flow model as input and replaces the resistances and inductances for vessels and junctions with values predicted by pretrained neural networks from the vessel or junction geometry.  See Rubio et. al. for more details.

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

The inputs are

* `anatomy` We provide four sets of pre-trained neural networks - one trained on aortic anatomies, one on aortofemoral anatomies, one on pulmonary anatomies, and one on a dataset containing all anatomies.  Users must choose which set to use.

* `zerod-json` This is the path to your original 0D input file, which will be modified to contain learned resistances and inductances.  This file may be generated from a 3D geometry using the `SimVascular` [ROM Simulation tool](https://simvascular.github.io/documentation/rom_simulation.html#tool).