# Official VideoMAEv2 model definition

`modeling_finetune.py` is an unmodified copy from [OpenGVLab/VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2/blob/29eab1e8a588d1b3ec0cdec7b03a86cca491b74b/models/modeling_finetune.py), commit `29eab1e8a588d1b3ec0cdec7b03a86cca491b74b`. The accompanying MIT licence is retained. Its file hash is pinned in `data/action_videomaev2.json`; the experiment refuses a changed implementation. No automatic remote-code loading is used.

The official model zoo points to `OpenGVLab/VideoMAE2`, `distill/vit_b_k710_dl_from_giant.pth`. The experiment pins revision `706cc172d65ebd4dedbee3f9c0183a93df9fa125`, loads tensor weights with `torch.load(weights_only=True)` and requires a strict state-dict match. The Hugging Face repository metadata declares Apache-2.0; source-code and checkpoint licences are separate.

Dependency: timm 1.0.22, installed on the experiment host into `.videomaev2-deps` with `--no-deps`, leaving its existing torch/torchvision environment unchanged. Model registration warnings for names already used by timm are expected; the experiment directly calls this module's factory, not the timm registry.

Input normalization uses ImageNet mean/std from the repository's actual video dataset pipeline (`dataset/datasets.py` at the same commit), rather than its unused generic `_cfg()` defaults. The dataset's spatial/temporal sampling is replaced deliberately by our fixed benchmark inputs. The official `forward_features` applies mean pooling followed by learned `fc_norm`; the K710 classifier is strictly loaded to validate the checkpoint but is excluded from embeddings.
