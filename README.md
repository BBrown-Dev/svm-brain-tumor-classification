# Brain Tumor Classification Using Support Vector Machines (SVM)
*A machine learning pipeline for MRI‑based tumor classification using HOG features, kernel comparison, and cross‑validated hyperparameter tuning.*

---

## Project Overview

This project implements a Support Vector Machine (SVM) classifier to detect and classify brain tumors from MRI images.  

The workflow includes:
- Image preprocessing  
- HOG feature extraction  
- Kernel comparison (Linear, Polynomial, & RBF)  
- Hyperparameter tuning using 5‑fold cross‑validation
- Performance evaluation  
- Decision boundary visualization using PCA  

The final model achieves 91% test accuracy, demonstrating strong generalization and effective feature representation.

---

## Dataset

The dataset contains MRI images labeled into three tumor classes:

- Glioma
- Meningioma
- Pituitary

Images were resized to 128×128, converted to grayscale, and processed using Histogram of Oriented Gradients (HOG) to extract texture‑based features.

Dataset source is included in the References section.

---

## Methodology

### 1. Preprocessing
- Images resized to 128×128
- Converted to grayscale
- Normalized for feature extraction

### 2. Feature Extraction (HOG)
HOG features were extracted using:

- 9 orientations  
- 8×8 pixel cells  
- 2×2 block normalization  

This transforms each MRI into a high‑dimensional texture descriptor suitable for SVM classification.

### 3. Kernel Comparison
Three SVM kernels were evaluated:

| Kernel | Accuracy | Notes |
|--------|----------|-------|
| **Linear** | 0.8646 | Strong baseline; tumors show partial linear separability |
| **Polynomial** | 0.7863 | Underfits; too rigid for MRI texture patterns |
| **RBF** | 0.8613 | Best non‑linear baseline; improves significantly after tuning |

### 4. Hyperparameter Tuning (5‑Fold CV)
Grid search was applied over:

- `C: {0.1, 1, 10}`
- `gamma: {scale, 0.01, 0.001}`

**Best model:**

```
C = 10
gamma = 'scale'
CV Accuracy = 0.9041
```

### 5. Evaluation Metrics
The best model was evaluated on a held‑out test set.

---

## Results

### Test Accuracy: 91%

### Classification Report

| Class | Precision | Recall | F1‑Score |
|-------|-----------|--------|----------|
| Glioma | 0.88 | 0.95 | 0.92 |
| Meningioma | 0.92 | 0.72 | 0.80 |
| Pituitary | 0.93 | 0.97 | 0.95 |

### Key Insights
- Pituitary and glioma tumors were classified with high confidence.  
- Meningioma was the most challenging class due to visual similarity to glioma.  
- The close match between CV accuracy (90.41%) and test accuracy (91%) indicates excellent generalization and minimal overfitting.

---

## Visualizations

### Confusion Matrix
Saved to:
```
results/confusion_matrix.png
```

### Decision Boundary (PCA‑Reduced 2D Space)
Saved to:
```
results/decision_boundary.png
```

These plots help illustrate model performance and class separability.

---

## How to Run the Project

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the full pipeline:
```
python main.py
```

3. View results in the `results/` directory.

---

## References

Bahauddin. (2021, August 27). Build an image classifier with SVM! Analytics Vidhya. https://www.analyticsvidhya.com/blog/2021/06/build-an-image-classifier-with-svm/ 

Chaudhary, A. (2023). Brain Tumor Detection (YOLOv8, YOLOv9, YOLOv11) [Dataset]. Kaggle. https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection 

Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20, 273–297. https://doi.org/10.1007/BF00994018 

Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05) (Vol. 1, pp. 886–893). IEEE. https://doi.org/10.1109/CVPR.2005.177 

GeeksforGeeks. (2025, November 4). Implementing different SVM kernels. GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/ 

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830. http://jmlr.org/papers/v12/pedregosa11a.html 