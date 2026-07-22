# Automated Fracture Detection in Rock Cores Using Convolutional Neural Networks

**SURGE Research Internship — Department of Civil Engineering, Indian Institute of Technology Kanpur**

Prince Kumar Agarwal, Kunwar Dev Vrat Singh

---

## Overview

Rock Quality Designation (RQD) is a fundamental geotechnical index used for rock mass classification in mining, tunneling, petroleum exploration, and foundation engineering. It is computed from the fraction of intact core pieces longer than 10 cm recovered per borehole run. Manual RQD logging from core-tray photographs is slow, labour-intensive, and prone to inter-logger variability.

This project trains a CNN to classify Small Square Images (SSIs) extracted from rock-core-tray photographs as **intact** or **non-intact**, as an automated first step toward RQD estimation. The pipeline covers everything end-to-end: automated row/tray parsing from raw HyLogger images, SSI extraction, manual labelling with the VGG Image Annotator (VIA), an augmentation pipeline, and a custom CNN tuned via Optuna hyperparameter search.

## Objective

- Automated pipeline: core-tray photograph → per-piece SSIs, with no manual cropping.
- Robust binary classifier to sort SSIs into **Intact** / **Non-Intact** categories.
- Handle class imbalance (intact heavily outnumbers non-intact) and support hard-negative mining.
- Benchmark on held-out val/test splits and inspect model failure modes.

## Methodology

1. **Core tray images** — high-resolution HyLogger images containing multiple rock core trays.
2. **Row & tray detection (edge-density projection)** — row separators identified by computing the absolute derivative of horizontal/vertical pixel projections, isolating edge-density peaks to slice trays into individual core rows.
3. **SSI extraction (224×224 patches)** — rows are further sliced into fixed-size Small Square Images with configurable overlap.
4. **Manual labelling (VIA tool)** — 6,780 SSIs labelled intact / non-intact using the VGG Image Annotator, later expanded to 13,628 after augmentation.
5. **Augmentation pipeline** — ±10° rotation, X/Y shifting, zoom, brightness shifts, CLAHE, and Gaussian noise/blur, applied to the training split only.
6. **CNN training + Optuna tuning** — a custom Conv2D → BatchNorm → ReLU → MaxPool architecture (with optional attention) trained with focal loss + label smoothing and class weighting; hyperparameters (filters, learning rate, batch size, dropout) tuned via Optuna Bayesian search; early stopping on validation macro-F1.
7. **Evaluation & interpretability** — confusion matrices, precision/recall/F1 per class, ROC/PR curves, and qualitative error analysis on misclassified SSIs.

## Results

| Split | Accuracy | Macro F1 |
|---|---|---|
| Validation (685 SSIs) | 94% | 0.93 |
| Test (685 SSIs, held-out) | 91% | 0.90 |

- The best model iteration correctly classified 438 intact and 203 non-intact cores on the test split, with a low false-negative rate on fracture detection.
- The model is deliberately conservative on non-intact precision (recall 82–95%) — it occasionally misses subtle fractures (recall 62–85%) but rarely flags intact rock as fractured, a safe-side bias that produces conservative (never dangerously over-optimistic) RQD estimates.

Full details, figures, and discussion are in [`poster/surge_poster_rock_fracture_cnn.pdf`](poster/surge_poster_rock_fracture_cnn.pdf).

## Repository Structure

```
.
├── notebooks/
│   └── CNN_rock_core_SSI_classification.ipynb   # End-to-end pipeline (Colab-based)
├── poster/
│   └── surge_poster_rock_fracture_cnn.pdf        # SURGE research poster
├── requirements.txt
└── README.md
```

> **Note on data:** The original HyLogger core-tray images, extracted SSIs, and VIA annotation JSONs were stored on Google Drive during development (paths in the notebook reference `/content/drive/...` from Google Colab) and are not included here due to size and dataset-access restrictions. The notebook is provided to document the full methodology and is not directly runnable without pointing the data paths at your own copy of the dataset.

## Notebook Contents

The notebook (`notebooks/CNN_rock_core_SSI_classification.ipynb`) walks through:

- Row/tray separation from raw HyLogger images via projection-derivative peak detection
- SSI extraction and metadata bookkeeping
- VIA-based labelling workflow and label-merging utilities
- Dataset augmentation (rotation, shift, zoom, brightness, CLAHE, Gaussian noise/blur)
- `tf.data` pipeline construction with class-balanced sampling
- Custom CNN architecture with focal loss + label smoothing
- Optuna hyperparameter search and final model training
- Evaluation: confusion matrices, classification reports, ROC/PR curves, misclassified-sample inspection

## Applications in Engineering

Automated fracture detection applies to mineral exploration, petroleum reservoir characterisation, and tunnelling. Automating intact vs. non-intact core logging accelerates geological logging and frees experts for complex lithological interpretation rather than manual hairline fracture counting.

## References

1. Deere, D. U. (1964). *Technical description of rock cores.* Rock Mech. Eng. Geol.
2. Ariba, K. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD.
3. He, K. et al. (2016). *Deep Residual Learning for Image Recognition.*
4. Dutta, A. & Zisserman, A. (2019). *The VGG Image Annotator (VIA).* ACM Multimedia.

## Acknowledgements

This work was carried out under the SURGE Program, Department of Civil Engineering, IIT Kanpur, with thanks to the project mentor and lab members for annotation guidance, and to NVCL / Geoscience Australia for HyLogger core-tray data.

## License

MIT — see [LICENSE](LICENSE).
