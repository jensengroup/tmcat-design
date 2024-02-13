Transition-Metal Catalyst Design for the Suzuki Reaction
========================================================

Code to the paper *Beyond Predefined Ligand Libraries: A Genetic Algorithm Approach for De Novo Discovery of Catalysts for the Suzuki Coupling Reactions*[^1].\
The code is based on [`catalystGA`](https://github.com/juius/catalystGA.git) version 1.0.

* [data](./data): Output data from GA runs
* [scripts](./scripts): Scripts to run GAs
* [smiles](./smiles): SMILES of ligands that are used as starting populations
* [suzuki](./smiles): Implementation of `SuzukiCatalyst`, based on [`BaseCatalyst`](https://github.com/juius/catalystGA/blob/1d62d0fbae784ad04fcc46a09529c38c958ad226/catalystGA/components.py#L26C8-L26C8)

## How to get started

> [!IMPORTANT]
> This code relies on some fixes that are not implemented in RDKit ≤ 2023.03.2.

1. Install the modified RDKit
<details>
  <summary>Details</summary>

  A modified version of RDKit can be downloaded from https://www.github.com/juius/rdkit/tree/custom \
  This version assigns hybridisation states of atoms from which dative bonds start correctly, the changes are described in the SI of the main paper[^1].\
  Exemplary steps to download and compile the modified RDKit in a virtual environment:


  ```sh
conda create -n suzuki python=3.9 -y
conda activate suzuki

conda install -y cmake cairo pillow eigen pkg-config
conda install -y boost-cpp boost
conda install -y -c anaconda py-boost
conda install -y gxx_linux-64
conda install -y numpy

git clone -b custom https://github.com/juius/rdkit.git
cd rdkit/

export RDBASE=`pwd`
export PYTHONPATH=${RDBASE}:${PYTHONPATH}
export LD_LIBRARY_PATH=${RDBASE}/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
echo "LD_LIBRARY_PATH: " $LD_LIBRARY_PATH
export QT_QPA_PLATFORM='offscreen'
export DYLD_FALLBACK_LIBRARY_PATH=${RDBASE}/lib

mkdir build && cd build
cmake -DPYTHON_INCLUDE_DIR="$CONDA_PREFIX/include/python3.9/" \
-DPYTHON_NUMPY_INCLUDE_PATH="$(python -c 'import numpy ; print(numpy.get_include())')" \
-DPy_ENABLE_SHARED=1 \
-DRDK_INSTALL_INTREE=ON \
-DRDK_INSTALL_STATIC_LIBS=OFF \
-DRDK_INSTALL_INTREE=OFF \
-DRDK_BUILD_CPP_TESTS=ON \
-DRDK_BUILD_PYTHON_WRAPPERS=ON \
-DRDK_BUILD_YAEHMOP_SUPPORT=ON \
-DRDK_BUILD_XYZ2MOL_SUPPORT=ON \
-DRDK_BUILD_CAIRO_SUPPORT=ON \
-DRDK_BUILD_INCHI_SUPPORT=ON \
-DRDK_BUILD_FREESASA_SUPPORT=ON \
-DBOOST_ROOT="$CONDA_PREFIX" \
-DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
-DINCHI_URL=https://rdkit.org/downloads/INCHI-1-SRC.zip \
-DBoost_NO_SYSTEM_PATHS=ON \
-DBoost_NO_BOOST_CMAKE=TRUE \
-DRDK_BOOST_PYTHON3_NAME="python39" \
..

make install -j 8

export PYTHONPATH="$RDBASE"
```

</details>

2. `git clone https://github.com/juius/suzuki.git`
3. `pip install path/to/suzuki/`
4. Install [`xtb`](https://xtb-docs.readthedocs.io/en/latest/setup.html) and [`orca`](https://www.orcasoftware.de/tutorials_orca/first_steps/install.html)

[^1]: Paper
