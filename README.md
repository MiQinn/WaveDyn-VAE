WaveDyn-VAE

Scaffold-based molecular generation with WavePDE

Overview

WaveDyn-VAE is a scaffold-based molecular generation framework that integrates variational autoencoders (VAEs) with WavePDE-guided latent space optimization for property-oriented molecular scaffold generation.

Repository Structure

checkpoints/: Pretrained model checkpoints.

data/: Dataset and intermediate data directories.

experiments/: Training and optimization scripts.

src/: Core model implementations.

result/: Output directory.

Installation

conda env create -f environment.yml
conda activate WaveDyn-VAE

Pretrained Models

Pretrained model checkpoints are available at GitHub Releases.

The release includes:

VAE checkpoint.

Property predictor checkpoint.

WavePDE checkpoints for different molecular property optimization objectives.

Download the required checkpoints and place them in the corresponding directories under checkpoints/.

Data Availability

The ChEMBL 35 dataset is not included in this repository and should be obtained separately.
