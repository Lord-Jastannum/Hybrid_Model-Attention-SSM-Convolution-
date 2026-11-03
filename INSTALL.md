# Installation (WSL2 Ubuntu, CUDA 12.8)

The supplied environment was tested by the project owner with Python 3.12,
PyTorch `2.11.0+cu128`, and `mamba_ssm` `2.3.2.post1+cu.12.8.torch.2.11`.
Keep that known-working environment initially; do not reinstall PyTorch unless a
specific compatibility error requires it.

From the project directory in WSL:

```bash
source mamba_env/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip install -r requirements.txt
python -m pip show mamba-ssm
python test_model.py
```

The environment must have a CUDA-compatible `mamba-ssm` installation. If you
need to recreate it, follow the official Mamba installation instructions for a
wheel compatible with the exact Python, PyTorch, and CUDA versions. Avoid
blindly installing a generic `mamba-ssm` source distribution, which may trigger
compilation or dependency changes.

Run the commands from WSL, not Windows PowerShell. The Linux environment is
`mamba_env/bin/activate`; `env/Scripts/activate` is a Windows virtualenv path.
