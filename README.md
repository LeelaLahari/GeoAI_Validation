# GeoAI Model Validation: Clay Foundation Model vs. Fine-Tuned ResNet50

Benchmarking a geospatial foundation model against a conventional fine-tuned CNN on Sentinel-2 imagery — and building the test infrastructure to catch the ways a model can be *wrong without looking wrong*.

## Why this project

Foundation models for satellite imagery are getting a lot of attention right now, and it's easy to assume "bigger pretraining = better everything." I wanted to actually test that assumption rather than take it on faith — specifically, whether the Clay geospatial foundation model (GFM) holds an edge over a plain fine-tuned ResNet50 once you control for training set size, spatial resolution, and pooling strategy.

The more interesting part of this project ended up being less "which model wins" and more "how do you know your evaluation pipeline isn't lying to you." Three separate times, my numbers looked fine on the surface and were actually wrong underneath. Catching those failures — not just the benchmark result — is most of what this repo documents.

## What's here

- Benchmark harness comparing Clay GFM and fine-tuned ResNet50 across **5 training set sizes, 3 spatial resolutions, and 2 pooling techniques** on 27,000 Sentinel-2 multispectral scenes
- A 15-case Pytest suite + CI pipeline covering band mapping, GSD (ground sample distance) scaling, and train/test leakage
- Notes on three silent inference failures I found mid-way through, and the assertions I added afterward so they can't happen again quietly

## Setup

| | |
|---|---|
| Dataset | 27,000 Sentinel-2 multispectral scenes |
| Training sizes tested | 5 (from small-sample to full dataset) |
| Resolutions tested | 3 |
| Pooling strategies | 2 |
| Models | Clay GFM (pretrained), ResNet50 (fine-tuned) |

```bash
# environment
pip install -r requirements.txt

# run the full benchmark sweep
python run_benchmark.py --config configs/sweep.yaml

# run just the test suite
pytest tests/ -v
```

## Test methodology

I didn't want to just train both models and report accuracy — a number like that hides more than it tells you. So before trusting any result, every run had to pass through:

1. **Band mapping checks** — making sure each model actually received the spectral bands I thought it did. Sentinel-2 band ordering is an easy thing to get silently wrong.
2. **GSD scaling checks** — resolution handling between the 3 configurations, to catch cases where resampling introduced artifacts the model could exploit without "really" learning anything useful.
3. **Train/test leakage checks** — scene-level and tile-level, since geospatial data loves to leak through spatial autocorrelation if you're not careful about your splits.

These became a 15-test Pytest suite that runs in CI on every change to the data pipeline or benchmark config — not a one-time sanity check, but something that has to keep passing.

## What I actually found

Three things broke quietly enough that a naive accuracy comparison would have missed them entirely:

- **Masking issue** — a subset of scenes were passing through with incomplete cloud/shadow masks, inflating apparent performance on "easy" tiles.
- **GSD mismatch** — one resolution configuration was feeding mismatched ground sample distances between the two models, meaning the comparison wasn't actually apples-to-apples at that resolution until I caught and fixed it.
- **Misaligned embeddings** — the most interesting one. A subtle coordinate misalignment during tiling caused a 9.6% label-agreement issue between the embedding space and the ground truth, which doesn't show up as an error — it just quietly drags your reported accuracy down (or up) depending on how the misalignment lands.

Each of these got a regression assertion added once fixed, so a future run can't reintroduce the same failure mode without the CI pipeline catching it.

## Results

After the pipeline was actually trustworthy, the comparison was consistent: fine-tuned ResNet50 outperformed the Clay GFM at every training size tested, including at full data (98.2% vs. 96.2%).

That's the opposite of what I expected going in. My read on why: a foundation model's advantage shows up most when labeled data is genuinely scarce (a handful of examples), and this task had enough labeled data that a smaller, task-specific fine-tuned model could just learn the specific distribution better. I'd expect the gap to favor Clay GFM to flip at much smaller training sizes — that's the natural next experiment.

## Limitations

- Single downstream task (land cover classification) — results may not generalize to segmentation or regression tasks on the same imagery
- Only one foundation model tested; results are specific to Clay GFM's pretraining, not foundation models generally
- Pooling strategies tested were limited to 2; architecture-specific pooling choices weren't explored

## Repo structure

```
.
├── configs/          # sweep and model configs
├── src/
│   ├── data/          # band mapping, GSD handling, leakage checks
│   ├── models/         # Clay GFM + ResNet50 wrappers
│   └── eval/           # benchmark + metrics
├── tests/             # the 15-case Pytest suite
└── notebooks/          # exploratory analysis, failure diagnosis
```

## Screenshots / evidence

*(Add here: CI pipeline passing screenshot, a before/after comparison on one of the three caught failures, and the final results table/plot.)*
