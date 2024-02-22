# Conformal QPM MPE Case Study
---

## Requirements:

In order to run this code, one must have the following installed on their system:

- Anaconda
- Java Runtime Version 17 or later
    - This is required for the STL robustness computation component

## Setup

Firstly, clone the repository and create a new Python 3.9 environment (with Anaconda)

```
$ git clone TODO LINK HERE
$ cd cqpm_mpe
$ conda create -n cqpm_mpe python=3.9
$ conda activate cqpm_mpe
```

Then, install the project-specific libraries:

1. Firstly, the Python requirements.
2. Secondly, the supplementary libraries included (note the **-e** flag).

These libraries have been bundled with the repository as changes have had to be made to make them compatible.

```
(cqpm_mpe) $ pip install -r requirements.txt
(cqpm_mpe) $ pip install -e .

(cqpm_mpe) $ pip install -e lib/pcheck
(cqpm_mpe) $ pip install -e lib/PettingZoo
(cqpm_mpe) $ pip install -e lib/tianshou
```

## To run

Firstly, change the root directory in the `path` section of `config.yaml`.

You can run the code by with the following options:

```
(cqpm_mpe) $ python cqpm_mpe --mode [train,generate,robust]
```
