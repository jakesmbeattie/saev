# Proposed Concrete Plan

Team: Fangxun, Henry (MS students), Jake (PhD, managing), Zach (domain expert), Sam (absent advisor)

## Background

We want to build SAE-generated dichotomous keys for Ohio's ~170 freshwater fish. The pipeline is: extract images, record ViT activations, train SAEs, explore features, map features to biological traits, and assemble a decision tree. See `brief.typ` for motivation.

We already have a working proof-of-concept: SAE run `um6hbn05` (DINOv3, TopK k=16) trained on FishVista demonstrates that SAEs recover fish anatomy (head, dorsal fin, eye) from unsupervised features. The question is whether this transfers to Ohio freshwater species and whether the features are useful for building a key.

## Overview

| Week | Fangxun | Henry | Zach |
|---|---|---|---|
| 1 | Download FishVista, record DINOv3 activations, train 40 SAEs | Extract fish from ToL-200M into ImageFolder, report coverage for ~170 Ohio species | Review fish-gallery.html, push on EPA data |
| 2 | Pareto analysis, inference, visuals, decision tree on FishVista SAEs | Record ToL activations, implement multi-shard training, launch combined sweep | Share gallery impressions at meeting |
| 3 | Pareto analysis on combined SAEs, gallery for Zach, experiment with SAE improvements | Build traits.parquet from FishBase, train sklearn classifiers on raw DINOv3, evaluate image cropping | Review combined-data gallery |
| 4 | Per-feature AUROC for species/genus/family, document interesting features | Train classifiers on SAE activations, compare to DINOv3 baselines, propose SAE improvements | Share gallery impressions at meeting |

## Organizational Notes

- I am deliberately suggesting that Fangxun and Henry have to collaborate aggressively week to week (Henry's extracted ToL data goes to Fangxun for training the next week, Fangunx's FishVista-trained SAES go to Henry for inference, etc.) because (1) I want to encourage cross-pollination of ideas and prevent you guys from siloing, even though it's harder to depend so tightly on each other and (2) I think getting a holistic understanding of the entire pipeline will make it possible for you to pursue parallel lines of research in the future from start to finish.
- I don't expect everything to get done every week. There are multiple intermediate checkpoints so that you can complete at least *something* every week, even if every TODO isn't completed.

## Week 1

| Person | Goals |
|---|---|
| @Zach | (1) Spend 20-30 minutes looking at fish-gallery.html. Write qualitative field notes and take screenshots of particularly interesting/weird things. Nothing is too small. (2) Push on the Ohio EPA electrofishing data. (3) Confirm the species list in `data/data.csv` and family list in `data/families.csv`. Are these the right ~170 species and ~26 families? Any missing, renamed, or outdated taxonomy? |
| @Sam | Get Henry access to OSC. |
| @Jake | Set up recurring weekly meeting(s). |
| @Fangxun | Train a sweep of TopK+AuxK SAEs on the FishVista ImgFolder |
| @Henry | Extract all fish images from ToL-200M into an ImageFolder on `/fs/scratch/PAS2136/saev-fish`. Use `contrib/freshwater_fish/scripts/extract_tol.py`. Then check: how many of the ~170 Ohio freshwater species from `data.csv` are present? Report per-species image counts. |

### Dev Environment Setup

On OSC, under your home directory, clone the fish fork of saev:

```sh
git clone git@github.com:Imageomics/saev-fish-dev.git
cd saev-fish-dev
uv sync
```

If you don't have `uv` installed, use https://docs.astral.sh/uv/getting-started/installation/. 

Then run `uv run launch.py -h` and you should see both "train" and "inference" subcommands.

### Save ViT Activations

```sh
uv run launch.py shards \
  --shards-root /fs/scratch/PAS2136/saev-fish/saev/shards \
  --family clip \
  --ckpt ViT-B-16/openai \
  --d-model 768 \
  --layers 11 \
  --content-tokens-per-example 196 \
  --batch-size 512 \
  --slurm-acct PAS2136 \
  --slurm-partition nextgen \
  data:cifar10 \
  --data.split train
```

This will save all ViT activations for a ViT-B/16 OpenAI CLIP model from layer 12/12 (11 is zero-indexed, so it is the 12th layer) to  `/fs/scratch/PAS2136/saev-fish/saev/shards` for the training split of CIFAR-10. 

**@Henry**: You want to basically set up a folder on `/fs/scratch/PAS2136/saev-fish/derived-datasets/tol200m-fish-v0.1` that has all the fish images, arranged by species, in a layout like:

```sh
/fs/scratch/PAS2136/saev-fish/datasets/tol200m-fish-v0.1/
  train/
    Animalia_Chordata_Actinopterygii_Aulopiformes_Synodontidae_Saurida_flamma/
      imagewhatever.jpg
      imagewhoever.jpg
      ...
    ...
  val/
    Animalia_Chordata_Actinopterygii_Aulopiformes_Synodontidae_Saurida_flamma/
      imagewhatever.jpg
      imagewhoever.jpg
      ...
    ...
```

Note that "modern phylogenetics views fish as a paraphyletic group that includes all vertebrates except tetrapods" ([source](https://en.wikipedia.org/wiki/Fish)) so there is no order or family that matches all fish, but they will be all in the *Chordata* phylum.

I think that `extract_tol.py` will be useful for this. 
Feel free to make changes and add new scripts under `contrib/freshwater_fish/scripts/`.

**@Fangxun**: Please train an SAE on the FishVista dataset. 
You'll have to download the dataset following these instructions: https://huggingface.co/datasets/imageomics/fish-vista#instructions-for-downloading-dataset-and-images.
Then you can process it with `contrib/trait_discovery/scripts/format_fishvista.py` which will create both ImageFolder and SegImageFolder. 
I would download to `/fs/ess/PAS2136/saev-fish/datasets/fish-vista` and then do the processing to `/fs/scratch/PAS2136/saev-fish/derived-datasets/fish-vista-imgfolder` and `/fs/scratch/PAS2136/saev-fish/derived-datasets/fish-vista-segfolder`.
Then you can record ViT activations using 

```sh
uv run launch.py shards \
  --shards-root /fs/scratch/PAS2136/saev-fish/saev/shards \
  --family dinov3 \
  --ckpt PATH_TO_DINOV3_VIT_L_CKPT.pth \
  --layers 21 23 \
  --d-model 1024 \
  --content-tokens-per-example 256 \
  --no-cls-token \
  --n-hours 4 \
  --batch-size 256 \
  --slurm-acct PAS2136 \
  --slurm-partition nextgen \
  data:img-folder \
  --data.root /fs/scratch/PAS2136/saev-fish/derived-datasets/fish-vista-imgfolder \
   --data.split training
```

And a similar command for the segfolder will work as well.
Then you can train an SAE using

```sh
uv run launch.py train --sweep contrib/freshwater_fish/exps/001-getting-started/sweeps/train.py --max-parallel 3
```

In `contrib/freshwater_fish/exps/001-getting-started/sweeps/train.py`, I would put:

```python
def make_cfgs() -> list[dict]:
    batch_size: int = 1024 * 16
    n_train = 100_000_000

    train_shards = "/fs/scratch/PAS2136/fish-saev/saev/shards/IMG_FOLDER_SHARDS"
    val_shards = "/fs/scratch/PAS2136/fish-saev/saev/shards/SEG_FOLDER_SHARDS"

    cfgs = []
    for layer in [21, 23]:
        for k in [16, 32, 64, 128]:
            for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
                cfgs.append({
                    "tags": ["001"],
                    "slurm_acct": "PAS2136",
                    "slurm_partition": "nextgen",
                    "n_hours": 8.0,
                    "lr": lr,
                    "n_lr_warmup": 500,
                    "n_sparsity_warmup": n_train // batch_size,
                    "runs_root": "/fs/ess/PAS2136/saev-fish/saev/runs",
                    "n_train": n_train,
                    "sae": {
                        "d_model": 1024,
                        "d_sae": 1024 * 32,
                        "normalize_w_dec": True,
                        "remove_parallel_grads": True,
                        "activation": {"top_k": k},
                        "reinit_blend": 0.8,
                    },
                    "train_data": {
                        "layer": layer,
                        "shards": train_shards,
                        "min_buffer_fill": 0.2,
                        "use_tmpdir": True,
                    },
                    "val_data": {
                        "layer": layer,
                        "shards": val_shards,
                        "use_tmpdir": True,
                    },
                })
    return cfgs
```

### Checklist

- [ ] Fangxun and Henry can both run `uv run launch.py -h` and see the help text.
- [ ] ImgFolder-compatible dataset of all fish species in ToL-200M in `/fs/scratch/PAS2136/saev-fish/derived-datasets/tol200m-fish-v0.1`
- [ ] Coverage report: how many of the ~170 Ohio freshwater species have images in ToL-200M? Per-species image counts. Which species are missing or have fewer than 10 images? If there are gaps, identify alternative image sources (iNaturalist, FishBase photos, museum collections) and start filling them in.
- [ ] ImgFolder-compatible FishVista dataset at `/fs/scratch/PAS2136/saev-fish/derived-datasets/fish-vista-imgfolder`
- [ ] ImgSegFolder-compatible FishVista dataset at `/fs/scratch/PAS2136/saev-fish/derived-datasets/fish-vista-segfolder`
- [ ] DINOv3 ViT-L/16 activations for the above ImgFolder datasets in `/fs/scratch/PAS2136/saev-fish/saev/shards`
- [ ] DINOv3 ViT-L/16 activations for the above ImgSegFolder datasets in `/fs/scratch/PAS2136/saev-fish/saev/shards`
- [ ] 40 SAEs trained on the ImgFolder split and evaluated on the ImgSegFolder split.

## Week 2

**@Zach**: Bring your initial impressions of the gallery to the weekly meeting and share any insights. 

**@Fangxun**: (1) Find the pareto-optimal SAEs. (2) Run inference on the pareto-optimal SAEs on the `fish-vista-segfolder` split. (3) Generate visuals for one pareto-optimal SAE to validate that your feature learning worked. (4) Train a decision tree classifier on all pareto-optimal SAEs, predicting the species in the `fish-vista-segfolder` images. There are examples of doing this in `contrib/trait_discovery/src/tdiscovery/classification.py` 

**@Henry**: (1) Record ViT activations on `tol200m-fish-v0.1`. 
(2) Update `src/saev/framework/train.py` to use multiple sets of training shards. There's a short spec and outline for how to do that in `contrib/freshwater_fish/docs/interal/specs/001-multiple-training-shards/`.
(3) Train a sweep of SAEs on a mix of `tol200m-fish-v0.1` and `fish-vista-imgfolder`.

### Checklist

- [ ] Pareto frontier plotted for week 1 FishVista SAE sweep.
- [ ] Inference completed on pareto-optimal SAEs using `fish-vista-segfolder`.
- [ ] Visuals generated for at least one pareto-optimal SAE.
- [ ] Decision tree classifier trained on pareto-optimal SAEs for FishVista species prediction. Report training accuracy.
- [ ] DINOv3 ViT-L/16 activations recorded for `tol200m-fish-v0.1`.
- [ ] Multi-shard training support merged into `src/saev/framework/train.py`.
- [ ] SAE sweep launched on combined `tol200m-fish-v0.1` + `fish-vista-imgfolder` shards.

## Week 3

### @Fangxun

1. Find pareto optimal SAEs from Henry's combined SAE training run.
2. Generate a visuals gallery for Zach. 
3. Compare the pareto frontier against the two runs. Which has a better reconstruction-sparsity tradeoff? What is your intuition for that? Suggest three ideas to train better SAEs and try them all, comparing reconstruction-sparsity tradeoffs AND visual galleries. Some ideas: higher resolution images, different ViT layers, different ViTs, wider SAEs, different kinds of SAEs (archetypal? ask Jake for inspiration on this).

### @Henry

1. Get species-level metadata for our fish. Write a script that pulls from various sources and writes a `/fs/ess/PAS2136/saev-fish/datasets/fish-metadata/traits.parquet` (or CSV) file with the following columns:

| Column | Source | Type | Required? |
|---|---|---|---|
| taxon | all (join key) | string | yes |
| common_name | FishBase | string | yes |
| class, order, family, genus, species | parsed from taxon | string | yes |
| body_shape | FishBase | string | nice to have |
| mouth_position | FishBase | string | nice to have |
| max_length_cm | FishBase | float | nice to have |
| trophic_level | FishBase | float | nice to have |
| habitat | FishBase | string | nice to have |
| n_tol_images | computed | int | yes |

The "nice to have" columns are whatever FishBase returns easily. Don't spend time chasing down missing values. The important thing is taxon (join key), the parsed taxonomy levels (for classification labels), and n_tol_images (sanity check). Everything else is just extra for downstream analysis.

2. Train sklearn classifiers on raw DINOv3 ViT activations. You can use a biobench-style training scheme ([link](https://github.com/samuelstevens/biobench)) or use the saved activations in combination with sklearn. I personally would use an OrderedDataloader in combination with `Config.tokens = 'special'` and just load all the [CLS] tokens into a big `np.ndarray`, and then pass it to the `.fit()` method on an sklearn classifier.
Note: I think you'll need to update OrderedDataloader to handle `Config.tokens = 'special'`, but I think the guides on the disk layout will make this straightfoward to both develop and test. 
3. Consider image data updates. Should we use [Moondream](https://moondream.ai/), [Isaac](https://www.perceptron.inc/demo), [SAM3](https://github.com/facebookresearch/sam3), etc to crop out backgrounds to produce fish-only images? Discuss the tradeoffs, the failures in SAE training so far, and whether you think it will be hard or complicated to achieve this.

### Checklist

- [ ] Pareto frontier plotted for combined (ToL + FishVista) SAE sweep. Side-by-side comparison with FishVista-only frontier.
- [ ] Written answer: does adding ToL data improve reconstruction? What is your hypothesis for why/why not?
- [ ] Gallery HTML shared with Zach for the best combined SAE.
- [ ] At least one experimental SAE idea launched (e.g., different resolution, wider SAE, different ViT). Pareto frontier plotted for that too.
- [ ] `traits.parquet` exists at `/fs/ess/PAS2136/saev-fish/datasets/fish-metadata/traits.parquet` with at least 150 species populated.
- [ ] Script that generates `traits.parquet` committed under `contrib/freshwater_fish/scripts/`.
- [ ] Species-level accuracy reported for at least one sklearn classifier (logistic regression or decision tree) on raw DINOv3 [CLS] tokens, predicting Ohio freshwater species. Report both training and validation accuracy.
- [ ] Written paragraph: what are the tradeoffs of cropping images (Moondream/Isaac/SAM3)? Should we do it? How hard is it?

## Week 4

### @Fangxun

1. For each SAE feature, which *species* activate it the most? Use AUROC in combination with SAE features and binary species labels.
2. Do the same with each family and genus as well.
3. Note down any features that are interesting! Take screenshots of visuals. 

For interesting visuals, I like to see the top examples of the feature on its positive class, the bottom examples of the feature on its positive class, and the top examples of the feature on its negative class. 

So for example, if you find that feature 1234 in an SAE has good AUROC for the [Latimeriidae](https://en.wikipedia.org/wiki/Latimeriidae) family, find the top 8 images of Latimeriidae that have a maximal SAE activation, find the top 8 images of *non*-Latimeriidae fish that have a maximal SAE activation, and find the 8 images of Latimeriidae with the *least* SAE activation. This helps with understanding precision (does it activate on non-Latimeriidae?) and recall (does it activate on all Latimeriidae?).

### @Henry

1. Train sklearn classifiers on the SAE activations. I think there are scripts for that in `contrib/trait_discovery`. Compare the results from classifiers on raw DINOv3 activations and SAE activations. What predictions are correct on both? What predictions are wrong on both? What predictions are different between the two?
2. Using Fangxun's AUROC analysis and the group's qualitative observations from weeks 1-3 (gallery reviews, interesting features, failure modes), manually select a subset of SAE features that seem biologically meaningful or visually interpretable. Train a new sklearn classifier on only these filtered features. Compare accuracy against the full-feature classifier from (1). The goal is to see whether a human-curated feature set can approach full-feature accuracy, which is a prerequisite for building a usable dichotomous key.
3. Investigate how to improve SAE training. This could be training on more relevant data (filter to specific species, filter to only specimen images instead of in-situ), getting more data, cropping images to the fish only (suggest last week), training SAEs for longer, using better learning rate scheduling, etc.

### Checklist

- [ ] Per-feature AUROC table for species, genus, and family. Saved as a parquet or CSV.
- [ ] At least 5 features with screenshots showing (a) top positives, (b) bottom positives, (c) top negatives. Shared with the group.
- [ ] Species-level accuracy reported for sklearn classifiers on all SAE activations, predicting Ohio freshwater species. Same metrics as week 3's raw DINOv3 classifiers.
- [ ] Hand-curated feature subset documented: which features were kept, why (AUROC, gallery observations, biological reasoning), and how many.
- [ ] Species-level accuracy reported for sklearn classifier on the curated SAE feature subset. Side-by-side comparison with full-feature and raw DINOv3 classifiers.
- [ ] Written comparison: which species/families does the SAE classifier get right that raw DINOv3 gets wrong, and vice versa?
- [ ] Written list of 2-3 concrete ideas for improving SAE training, with justification from the results so far.

## Technical Notes

- `make_cfgs()` and Python-based config files: for sweeps of SAE runs, instead of defining many TOML files, or a list of YAML files, you write a Python file with at least one function defined: `make_cfgs()`, which returns a `list[dict]`. These dicts are parsed into whatever `Config` class for the script you're running, and then typically submitted as a job array to Slurm.
- Inference outputs: `token_acts.npz` is a scipy sparse CSR matrix, not a dense array. `metrics.json` has normalized MSE. These run-specific outputs live under `.../runs/RUN_ID/inference/SHARD_HASH/`.
- Shard dataloaders: `ShuffledDataLoader` for training (reservoir buffer, global shuffle), `OrderedDataLoader` for inference (deterministic, sequential). Slightly different config interfaces. 
- ViT layer indexing: Indexing is 0-indexed. Layer 21 is the 22nd layer, layer 23 is the 24th layer.
- Parallel training: `train.py` support parallel training (see [this link](https://osu-nlp-group.github.io/saev/api/users/guide/#sweeps)). When a sweep has configs that differ in certain fields (train_data, sae.d_sae, etc.), they get split into separate Slurm jobs automatically. Configs that only differ in lr, k, or seed can share a job. This affects how many jobs get submitted and how long the queue wait is. You'll need to use `--max-parallel` in order to not train 20 SAEs on the same node.
