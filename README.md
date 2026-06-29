# GTZAN — Music Genre Classification

This repository implements a complete pipeline for music-genre recognition on the GTZAN dataset. It contains preprocessing code (MFCC and spectrogram extraction), several model implementations (MFCC-MLP baseline, Attention-CRNN, ViT, HPSS+Chroma fusion, ResNet baseline) and end-to-end experiments including augmentation, ensembling and explainability (Grad-CAM, SHAP/LIME hooks).

**What this repo contains**

- **Notebook**: [preprocessed_dataset.ipynb](preprocessed_dataset.ipynb) : Dataset download, audio segmentation, MFCC extraction, mel-spectrogram creation and export to reproducible `.npy` files.
- **Notebook**: [music_genre_classification_mfcc.ipynb](music_genre_classification_mfcc.ipynb) : MFCC-based baseline (MLP) training, evaluation and quick visualization of predictions.
- **Notebook**: [Synergistic_Spectral_Temporal_Attention_and_Multi_Model_Stacked_Ensembling_for_Robust_Music_Genre_Recognition.ipynb](Synergistic_Spectral_Temporal_Attention_and_Multi_Model_Stacked_Ensembling_for_Robust_Music_Genre_Recognition.ipynb) : Full research notebook — configuration, data loading (preprocessed or raw), multi-model architectures (Attention-CRNN, ViT, MFCC-MLP, fusion, ResNet), augmentation (SpecAugment / Mixup), training, evaluation, ensembling, XAI and deployment helpers (Gradio, TFLite, pruning).

**Quick summary**

- **Dataset**: GTZAN (10 genres, 100 tracks/genre, 30s each). The notebooks expect the original GTZAN folder structure: `Data/genres_original` (download via Kaggle).
- **Preprocessing**: audio → segments (default: 10 segments per 30s track → 3s each) → features: MFCC (13 or 40 coeffs mean per segment) and 128×128 mel-spectrogram segments (saved as `.npy` arrays). A single corrupted file (`jazz.00054.wav`) is intentionally skipped in extraction.
- **Models**: lightweight MFCC-MLP baseline plus a stronger Attention-CRNN (3 CNN blocks → BiGRU → temporal multi-head attention → attention pooling). Additional baselines: ViT, HPSS+Chroma fusion (3-channel), ResNet18-light, SVM (MFCC baseline). Ensembling: average, weighted, stacking with logistic regression.

**Reproduce: recommended steps**

1. Install dependencies (recommended in a virtualenv or Colab):

```bash
pip install numpy scipy librosa scikit-learn tqdm matplotlib seaborn tensorflow opendatasets
# Optional: for XAI or demo
pip install shap lime gradio networkx
```

2. Download the GTZAN dataset (Kaggle). Use the `preprocessed_dataset.ipynb` notebook which includes an automated `opendatasets` command. Alternatively, download manually and place the files under `Data/genres_original`.

3. Preprocess and save arrays (run the preprocessing notebook):

- Open [preprocessed_dataset.ipynb](preprocessed_dataset.ipynb) and run cells to extract MFCCs and mel-spectrogram segments. The notebook saves:

	- `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy` (MFCC-derived arrays)
	- `X_train_images.npy`, `X_test_images.npy`, `y_train_images.npy`, `y_test_images.npy` (mel-spectrogram image arrays)
	- `genre_classes.npy`

4. Run experiments:

- For a quick baseline, run [music_genre_classification_mfcc.ipynb](music_genre_classification_mfcc.ipynb) (loads `.npy` MFCC arrays and trains an MLP).
- For full research experiments (attention CRNN, ViT, fusion, ensembling, XAI), run [Synergistic_Spectral_Temporal_Attention_and_Multi_Model_Stacked_Ensembling_for_Robust_Music_Genre_Recognition.ipynb](Synergistic_Spectral_Temporal_Attention_and_Multi_Model_Stacked_Ensembling_for_Robust_Music_Genre_Recognition.ipynb). Important configuration flags are at the top of the notebook (GPU recommended): `USE_PREPROCESSED`, `USE_SPECAUGMENT`, `USE_MIXUP`, `EPOCHS`, `BATCH_SIZE`.

**Notes & reproducibility tips**

- The large notebook prefers preprocessed `.npy` arrays for speed (`USE_PREPROCESSED = True`). If these are missing it will extract mel-spectrograms from raw audio automatically.
- The code segments the 30s tracks into 3s segments by default (`NUM_SEGMENTS = 10`). Keep segmentation consistent when comparing models.
- When ensembling across models trained on different feature sets, the notebooks use a common subset (200 samples) to ensure compatible prediction shapes — see the ensemble section for details.
- Checkpoint files and example exports are created by the notebooks, e.g. `best_attention_crnn.keras` and `crnn_quantized.tflite` (if quantized export is enabled).

**File map**

- [preprocessed_dataset.ipynb](preprocessed_dataset.ipynb) — preprocessing and `.npy` export.
- [music_genre_classification_mfcc.ipynb](music_genre_classification_mfcc.ipynb) — MFCC baseline workflow.
- [Synergistic_Spectral_Temporal_Attention_and_Multi_Model_Stacked_Ensembling_for_Robust_Music_Genre_Recognition.ipynb](Synergistic_Spectral_Temporal_Attention_and_Multi_Model_Stacked_Ensembling_for_Robust_Music_Genre_Recognition.ipynb) — full pipeline, models and analysis.

**Suggested next steps**

- Create a `requirements.txt` (pin key package versions) so results are reproducible.
- Optionally add a small `scripts/` folder to run preprocessing and training from the command line (for example `python -m experiments.train --config configs/attention_crnn.yaml`).

**License & contact**

- If you plan to publish this project, add a `LICENSE` file to clarify reuse terms.
- For questions or collaboration, please open an issue in this repository.
