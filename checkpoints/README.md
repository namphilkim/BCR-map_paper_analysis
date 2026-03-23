# Checkpoints directory

PyTorch **`.ckpt` weights are not stored in this Git repository**. They are archived separately on **Zenodo** (see the main [README.md](../README.md) for the DOI and download instructions).

After you download and extract the Zenodo archive, this folder should mirror the layout used in the paper, for example:

- `checkpoints/In-house/...`
- `checkpoints/Mal-ID/...`

Training and evaluation scripts resolve paths relative to the repository root, so keep that structure when unpacking.
