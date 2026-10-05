# LC-Rec tokenizer source

Copied model files only, unchanged, from https://github.com/RUCAIBox/LC-Rec
commit `e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669`:

- `index/models/layers.py`
- `index/models/vq.py`
- `index/models/rq.py`
- `index/models/rqvae.py`

These are reusable quantizer model components. Model dimensions and codebook
sizes are constructor arguments. The release construction entry point is
`semicd build-codebook --method rqvae`. The historical experiment runner
and its fixed configurations are not part of this release tree.

No recommendation model, patient text or pretrained RQ-VAE weights are copied.
