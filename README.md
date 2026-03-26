This package takes a 0D cardiovascular flow model as input and replaces the resistances and inductances for vessels and junctions with values predicted by pretrained neural networks from the vessel or junction geometry.  See Rubio et. al. for more details.

The following command installs this package and its dependencies:

```bash
conda create -n learnedzerod python=3.11 -y && \
conda activate learnedzerod && \
pip install -e ".[dev]"
```


```bash
learned-zerod\
  --anatomy aortic \ # Type of anatomies the applied neural networks were trained on.  Choose from "aortic", "aortofemoral", "pulmonary" or "all"
  --zerod-json /path/case.json \ # Path to your original 0D input file (generate with SimVascular)
  --centerline-vtp /path/unsteady_soln.vtp \ # Path to geometry centerline  (generate with SimVascular)
  --svzerod /path/to/svZeroDSolver \  # Path to your installation of svZeroDSolver
  --output-dir /path/to/out # Path to output directory 
```

