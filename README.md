**WaveDyn-VAE**

Scaffold-based molecular generation with WavePDE

***Overview***

WaveDyn-VAE is a scaffold-based molecular generation framework that integrates variational autoencoders (VAEs) with WavePDE-guided latent space optimization to generate molecular scaffolds.

***Repository Structure***
- `checkpoints/`: Pretrained model checkpoints.
- `data/`: Dataset and intermediate data directories.
- `experiments/`: Training and optimization scripts.
- `src/`: Core model implementations.
- `result/`: Intermediate and generated results.

***Installation***

conda env create -f environment.yml
conda activate WaveDyn-VAE

Pretrained model checkpoints are available at [GitHub Releases](https://github.com/MiQinn/WaveDyn-VAE/releases).

The release includes:
- VAE checkpoint: `vae_chembl_sca_screened.pt`
- Property predictor checkpoint: `prop_predictor_random_39891_seed42_scaffold.pt`
- WavePDE checkpoints for different molecular property optimization objectives: `wavepde_random_*.pt`

***Model placement***

After downloading the VAE and WavePDE weights, rename each file to `checkpoint.pt` and place it in the corresponding directory:
1. VAE checkpoint:
   `checkpoints/vae/chembl_sca_screened/checkpoint.pt`
2. WavePDE checkpoints:
   - `checkpoints/wavepde_prop/random/fsp3/checkpoint.pt`
   - `checkpoints/wavepde_prop/random/logp/checkpoint.pt`
   - `checkpoints/wavepde_prop/random/n_atoms/checkpoint.pt`
   - `checkpoints/wavepde_prop/random/n_ring/checkpoint.pt`
   - `checkpoints/wavepde_prop/random/n_star/checkpoint.pt`
3. Property predictor:
   `checkpoints/prop_predictor/prop_predictor_random_39891_seed42_scaffold.pt`
The property predictor and structure predictor checkpoints are provided in the repository.

***Data Availability***

The ChEMBL 35 dataset is not included in this repository and should be obtained separately. Other datasets used for model training and molecule generation should be prepared according to the corresponding scripts.

***Workflow***

The molecular generation workflow consists of the following steps:
1、*Scaffold extraction:* Sca_bondbreak_dummy.py extracts molecular scaffolds and introduces dummy atoms (*) at the attachment 、points.
2、*VAE training:* train_vae.py trains the variational autoencoder.
3、*Predictor data preparation:* prepare_random_data.py prepares training datasets for property and structural predictors.
4、*Property predictor training:* train_prop_predictor.sh trains the molecular property predictors.
5、*Structure predictor training:* train_structure_predictor.py trains the structural predictor.
6、*WavePDE training:* train_wavepde_prop.sh trains property-guided WavePDE models based on the pretrained predictors.
7、*Scaffold optimization and generation:* optimization.sh performs latent space optimization and decodes the optimized representations into new scaffolds.
8、*Anchor library construction:* build_supervised_anchor_libiary.py constructs libraries of sidechains and scaffold attachment points.
9、*Combination enumeration:* enumerate_supervised_intermediate.py enumerates scaffold–sidechain combinations based on the constructed libraries.
10、*Molecular reconnection:* reconnect_suppervised_assigenments.py reconnects the enumerated scaffolds and sidechains to generate complete molecules.
11、*Molecular screening:* The generated molecules are filtered based on structural and physicochemical criteria to obtain candidate molecules.
