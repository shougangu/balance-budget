# verl venv setup (Alliance clusters)

Pinned: **verl 0.9.0**, venv at `/project/6105902/shougan/venvs/verl-0.9.0`
(project FS so every cluster's nodes see it; never scratch — 60d purge).
The resolved dependency matrix after install lives beside it in
`resolved-requirements.txt`.

## The rules that make installs work here

1. **Never `pip install -U pip`.** The venv inherits the cluster-patched pip
   (23.2.1) from the cvmfs python module; that patch is what resolves
   `+computecanada` wheels and `*-noinstall` shim requirements. Stock pip fails
   with `vllm has an invalid wheel, .dist-info directory not found`.
2. **`module load gcc arrow/19.0.1 opencv` before using the venv.** The computecanada
   wheels lean on cluster modules: vllm's `opencv-noinstall` and datasets'
   pyarrow are both dummy sdists that error unless the matching module provides
   the import (each dummy's error message names its module). Pin the arrow
   version: an unversioned `module load arrow` can resolve to a different
   release than the placeholder dist below claims.
3. `PIP_CONFIG_FILE` already points at the computecanada wheelhouse config;
   leave it. Set `TMPDIR=/scratch/$USER/tmp` for multi-GB wheel unpacks
   (home is small).
4. The wheelhouse's vllm can fail to install *through the resolver* even when
   the wheel file itself is valid; the workaround is download-then-install:
   `pip download --no-deps vllm==<ver> -d <dir>` then `pip install <dir>/<wheel>`.
5. **Some requirements (vllm's `opencv-python-headless>=4.13.0`, verl's
   `pyarrow>=19.0.0`) have no installable candidate here** — the modules provide
   the import at runtime but not an in-venv dist, the wheelhouse only has
   deliberate-error dummy candidates (versions 9999, 25.0.0+dummy, 4.9999...),
   and CC pip rejects PyPI manylinux wheels. Two-part fix:
   (a) register a placeholder dist per package so the requirement reads satisfied:
   ```bash
   D=<venv>/lib/python3.11/site-packages/opencv_python_headless-4.13.0.dist-info
   mkdir -p $D
   printf 'Metadata-Version: 2.1\nName: opencv-python-headless\nVersion: 4.13.0\n' > $D/METADATA
   printf 'placeholder\n' > $D/INSTALLER; touch $D/RECORD
   # same shape for pyarrow-19.0.1.dist-info (matching the arrow module's version)
   ```
   (b) pass `-c tuning/verl/cc-constraints.txt` on every pip install into this
   venv: the resolver otherwise *explores* the higher-versioned dummy candidates
   during multi-package resolves and dies building their `*-noinstall` metadata,
   even with the requirement already satisfied.
6. **Run the install inside a CPU Slurm job**, not on a login node — the login
   nodes' resource limits SIGKILL pip mid-unpack of the vllm wheel. Killarney
   compute nodes have internet.

## Recipe

```bash
module load gcc opencv
/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/python/3.11.5/bin/python3 \
    -m venv /project/6105902/shougan/venvs/verl-0.9.0
V=/project/6105902/shougan/venvs/verl-0.9.0/bin
export TMPDIR=/scratch/$USER/tmp
$V/pip download --no-deps -d $TMPDIR/wheelprobe "vllm>=0.18.0"
# (register the opencv placeholder from rule 5 here)
$V/pip install $TMPDIR/wheelprobe/vllm-*.whl
# Base verl, NOT verl[vllm]: the extra makes the resolver scan the wheelhouse's
# broken vllm candidates even with vllm already installed. tensordict carries
# the extra's only other pin.
C=tuning/verl/cc-constraints.txt
$V/pip install -c $C "verl==0.9.0" "tensordict>=0.8.0,!=0.9.0,<0.10.1"
# The wheelhouse vllm (0.25) pins torch 2.11 / torchvision 0.26 / torchaudio 2.11,
# which the download-then-install step above does not enforce; an unpinned
# flash_attn resolves to the +torch29 build and drags torch down to 2.9.1, and a
# torch-2.9 core under a torch-2.11 torchvision fails at import
# ("operator torchvision::nms does not exist"). Pin the whole 2.11 stack.
$V/pip install -c $C "torch==2.11.0" "flash_attn==2.8.3+torch211.computecanada"
# tensordict 0.10.0 (verl caps it <0.10.1) declares torch~=2.9; pip warns, the
# import works on 2.11 (probe job 5160704).
# Reward grader, same version as the eval venv so reward and metric agree.
$V/pip install -c $C "math-verify==0.9.0"
$V/pip freeze > /project/6105902/shougan/venvs/verl-0.9.0/resolved-requirements.txt
```

The repo itself is imported via PYTHONPATH (the sbatch script exports it);
`tuning.verl.*`, `tuning.evaluation.math_scoring`,
`tuning.training.passk.decisions`, and
`tuning.training.pipeline.checkpoint_metadata` are the only repo modules the
verl venv touches — all free of trl/unsloth/eval-stack imports (the passk
package exports its callback lazily for exactly this reason).

## Version coupling

`tuning/verl/budget_trainer.py` subclasses `RayPPOTrainer.fit` from the pinned
release. On any verl upgrade, re-diff the upstream `fit()` against the subclass
before trusting it.
