# Quantum Reservoir Computing in the Whillans Ice Sheet

Quantum reservoir computing classical-hybrid model for icequake time-to-next-slip prediction in Antarctic ice sheet.

## Environment Setup

Using `conda`, create an environment from `environment.yml` using the command

```bash
conda create -f environment.yml
```

## Running Pipelines

To run the classical and noiseless pipelines, run the jupyter notebooks in their respective folders. To run the noisy pipeline, details are in both of the shell scripts in the noisy folder, but a HPC cluster set up with SLURM must be available. To run the pipeline which leverages IBM hardware, an IBM quantum runtime API key is required.

### Random Seed Note

Due to the varying nature of computer architectures, exactly the same random number seeds can give way to different results on different computers. As such, results may vary slightly from computer to computer.
