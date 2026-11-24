# UrbanXAI
### Explainable Satellite-Image-Based Urban Building Analysis

UrbanXAI is a deep-learning-based system for building footprint segmentation and explainable urban analysis using satellite imagery from the SpaceNet 2 Paris dataset.

The project combines semantic segmentation, visual explainability, urban statistics, and an interactive dashboard.

## 1. Project Objectives

- Detect and segment building footprints from satellite images.
- Visualize model predictions using Grad-CAM.
- Analyse prediction sensitivity using SHAP-style patch occlusion.
- Calculate a preliminary Urban Explanation Consistency Score (UECS).
- Estimate building counts, built-up percentage, and building density.
- Present results through visualizations and an interactive dashboard.

## 2. Technology Stack

- Python
- TensorFlow / Keras
- EfficientNetB3
- Attention U-Net
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- HTML, CSS and JavaScript

## 3. Dataset

**Dataset:** SpaceNet 2 – Building Detection  
**Study Area:** Paris, France  
**Imagery:** WorldView-3 RGB Pan-Sharpened satellite imagery.

Dataset source:
https://www.kaggle.com/datasets/amerii/spacenet-2-building-detection-v2

Dataset split:

| Split | Images |
|---|---:|
| Training | 803 |
| Validation | 172 |
| Testing | 173 |
| Total | 1,148 |

Images and corresponding building masks are resized to 128 × 128 pixels.

The original dataset and preprocessed arrays are not included in this repository because of their size.

## 4. Model Architecture

The primary segmentation model uses an EfficientNetB3 encoder with an Attention U-Net decoder.

The encoder extracts hierarchical image features, while the decoder reconstructs spatial information to produce pixel-level building segmentation masks.

Attention mechanisms help the network focus on relevant spatial regions.

## 5. Model Performance

The best evaluated model achieved the following test-set results at a prediction threshold of 0.475:

| Metric | Result |
|---|---:|
| IoU | 60.02% |
| Dice Score | 75.02% |
| Precision | 76.98% |
| Recall | 73.15% |
| F1 Score | 75.02% |

These results represent the evaluated performance of the selected model on the held-out test set.

## 6. Explainability

### Grad-CAM

Grad-CAM generates heatmaps highlighting image regions associated with the model's predictions.

### Patch Occlusion

Patch occlusion analyses how predictions change when individual image regions are masked.

It provides a visual representation of prediction sensitivity to different parts of the satellite image.

Both techniques are used to examine model behaviour. Their visualizations should not be interpreted as proof of prediction correctness.

## 7. Urban Explanation Consistency Score (UECS)

UECS is a preliminary project-specific measure used to examine spatial similarity between the Grad-CAM and occlusion explanation maps.

The score is calculated using:

UECS = (Pearson Correlation + 1) / 2

Preliminary results:

| Metric | Result |
|---|---:|
| Mean UECS | 0.4843 |
| Valid explanation pairs | 17 / 20 |
| Coverage | 85% |

Three pairs were excluded because of constant explanation maps.

UECS measures similarity between two explanation methods. It does not establish explanation faithfulness or correctness.

## 8. Urban Statistics

Urban statistics were calculated using 172 validation images.

| Metric | Result |
|---|---:|
| Predicted building components | 1,848 |
| Ground-truth components | 2,153 |
| Predicted built-up percentage | 6.3532% |
| Ground-truth built-up percentage | 6.7498% |
| Mean predicted component density | 6.5577 per 10,000 pixels |

Building counts are estimated using connected components. Touching buildings may be merged into a single component.

These are image-based estimates and are not geospatial measurements of real-world building area.

## 9. Interactive Dashboard

The UrbanXAI dashboard presents:

- Segmentation results
- Model evaluation metrics
- Grad-CAM visualizations
- Occlusion explanations
- UECS analysis
- Urban statistics
- Graphical comparisons

The dashboard and selected figures are intended to provide an accessible overview of the project outputs.

## 10. Important Source Files

| File | Description |
|---|---|
| `train_model.py` | Main model training implementation |
| `preprocessing.py` | Satellite image and mask preprocessing |
| `evaluate_saved_model.py` | Evaluation of a saved model |
| `evaluate_experiment1.py` | Experiment 1 evaluation |
| `generate_gradcam.py` | Grad-CAM generation |
| `generate_occlusion.py` | Patch occlusion generation |
| `calculate_uecs.py` | UECS calculation |
| `calculate_urban_statistics.py` | Urban statistics calculation |
| `explainability_report.py` | Explainability reporting |
| `create_styled_figures.py` | Figure and dashboard generation |

Additional experiment-specific scripts are available in the repository.

## 11. Project Structure

```text
UrbanXAI/
├── models/
├── training/
├── explainability/
├── urban_statistics/
├── dashboard/
├── reports/
├── README.md
├── requirements.txt
└── .gitignore
eof
