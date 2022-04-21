# Feature Selection with Quantum Computing

This repository contains the source code for the article "Towards Feature Selection for Ranking and Classification Exploiting Quantum Annealers" published at SIGIR 2022. See the websites of our [quantum computing group](https://quantum.polimi.it/) for more information on our teams and works.

Here we explain how to install dependencies, setup the connection to D-Wave Leap quantum cloud services and how to run
experiments included in this repository.

## Installation

> NOTE: This repository requires Python 3.8 and has been developed for Linux

It is suggested to install all the required packages into a new Python environment. So, after repository checkout, enter
the repository folder and run the following commands to create a new environment:

If you're using `conda`:

```console
conda create -n QFeatureSelection python=3.8 anaconda
conda activate QFeatureSelection
```

>If you run the experiments on the terminal it may be necessary to add this project in the PYTHONPATH environmental variable:
>```console
>export PYTHONPATH=$PYTHONPATH:/path/to/project/folder
>```

Then, make sure you correctly activated the environment and install all the required packages through `pip`:

```console
pip install -r requirements.txt
```


## D-Wave Setup

In order to make use of D-Wave cloud services you must first sign-up to [D-Wave Leap](https://cloud.dwavesys.com/leap/)
and get your API token.

Then, you need to run the following command in the newly created Python environment:

```console
dwave setup
```

This is a guided setup for D-Wave Ocean SDK. When asked to select non-open-source packages to install you should
answer `y` and install at least _D-Wave Drivers_ (the D-Wave Problem Inspector package is not required, but could be
useful to analyse problem solutions, if solving problems with the QPU only).

Then, continue the configuration by setting custom properties (or keeping the default ones, as we suggest), apart from
the `Authentication token` field, where you should paste your API token obtained on the D-Wave Leap dashboard.

You should now be able to connect to D-Wave cloud services. In order to verify the connection, you can use the following
command, which will send a test problem to D-Wave's QPU:

```console
dwave ping
```

## PyMIToolbox Setup

PyMIToolbox is a Python wrapper to the C library MIToolbox which is used to compute Mutual Information.

### Download
In order to use PyMIToolbox you first need to download and compile the MIToolbox library in the PyMIToolbox directory.
To download the MIToolbox source code execute the following command:

```console
cd PyMIToolbox/
wget https://github.com/Craigacp/MIToolbox/archive/refs/tags/v3.0.2.zip
```

Unzip the file with:
```console
unzip v3.0.2.zip
```
and rename the extracted folder with:
```console
mv MIToolbox-3.0.2 MIToolbox
```

### Building the C library
Now, go into the MIToolbox directory and compile the C library.
If you are on Linux or macOS run the following command:
```console
cd MIToolbox/
make x64
```
while on Windows, install [MinGW](https://sourceforge.net/projects/mingw-w64/), add MinGW binaries to the PATH and run:
```console
make x64_win
```

This will result in a compiled library file (`.so` on Linux/macOS and `.dll` on Windows) to be placed in the 
`PyMIToolbox/MIToolbox/` folder. If you don't see the file, it may have been compiled to another directory and should be moved to the correct folder.


## Running Classification Experiments

To run the experiments enter the root folder of the project, activate the environment and run the following script:

```console
conda activate QFeatureSelection
python run_feature_selection.py
```

This python script will automatically download and split the datasets used in the experiments.
The resulting splits are saved in the `results_classification/[dataset_name]/data` directory.

The script will then proceed to run all experiments: baseline and QUBO with both classical and quantum based solvers.
All the results will be saved in the `results_classification/[dataset_name]/[method_name]` directory. 

> NOTE: Running all the experiments requires a significant amount of QPU time and will exhaust all the free time given with the developer plan on
> D-Wave Leap. If the available time runs out it will result in errors or invalid selections.
> We suggest to select a limited number of datasets at a time.

For each dataset the script will also generate a dataframe summarizing the results and at the end of all experiments it will generate summary tables in latex format.

Within each dataset folder the file `result_dataset_summary.csv` will contain one row per each feature selection method and, for QUBO methods, QUBO solvers. The row is selected as the one with the best validation score across all the target numbers of features (i.e., k) for that experiment.

## Running Ranking Experiments

To run the Ranking experiments on LETOR, you first need to download the datasets:

> [Downoad OHSUMED](https://onedrive.live.com/?authkey=%21ACnoZZSZVfHPJd0&cid=8FEADC23D838BDA8&id=8FEADC23D838BDA8%21144&parId=8FEADC23D838BDA8%21107&o=OneUp)
> 
> [Download MQ2007](https://onedrive.live.com/?authkey=%21ACnoZZSZVfHPJd0&cid=8FEADC23D838BDA8&id=8FEADC23D838BDA8%21120&parId=8FEADC23D838BDA8%21107&o=OneUp)
> 
> [Download MQ2008](https://onedrive.live.com/?authkey=%21ACnoZZSZVfHPJd0&cid=8FEADC23D838BDA8&id=8FEADC23D838BDA8%21114&parId=8FEADC23D838BDA8%21107&o=OneUp)

After downloading, unzip them in the folder `data/letor/` in order to have the structure `data/letor/[dataset_name]`.

### RankLib

In order to execute the Learning to Rank algorithm you need the [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib%20Installation/) library.
In the experiments RankLib 2.17 is used, you can download it [here](https://sourceforge.net/projects/lemur/files/lemur/RankLib-2.17/RankLib-2.17.jar/download).
Place the downloaded `RankLib-2.17.jar` file in the `RankLib/` directory.

### Running

To run the experiments enter the root folder of the project, activate the environment and run the following script:

```console
conda activate QFeatureSelection
python run_letor_feature_selection.py
```

The script will proceed to run all experiments: baseline and QUBO with both classical and quantum based solvers.
All the results will be saved in the `results_ranking/[dataset_name]/[method_name]` directory. 

> NOTE: Running all the experiments requires a significant amount of QPU time and will exhaust all the free time given with the developer plan on
> D-Wave Leap. If the available time runs out it will result in errors or invalid selections.
> We suggest to select a limited number of datasets at a time.

For each dataset the script will also generate a dataframe summarizing the results.
Within each dataset folder the file `result_dataset_summary.csv` will contain (slightly differently from classification experiments) all the feature selection information.

The results of the ranking algorithm will be instead saved in the `results_ranking/processed/[dataset_name]_eval.csv` files.

> Note that the ranking experiments have been tested only on a Unix system with `bash`.
> Running them on Windows may require additional setup.
