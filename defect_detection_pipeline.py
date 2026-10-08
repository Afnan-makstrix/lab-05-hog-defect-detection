"""
Laboratory 05: HOG-Based Industrial Surface Defect Detection & Classification
Student Name: Afnan Khan (@Afnan-makstrix)
Course: Computer Vision & Industrial Quality Automation
Benchmark: NEU Surface Defect Database (Hot-Rolled Steel Strips)

Functional Architecture:
  1. Corpus Ingestion & Standardization (128x128 Monochromatic)
  2. HOG Feature Signature Extraction & Spatial Orientation Mapping
  3. Multi-Model Classifier Suite: Linear SVM vs Random Forest vs K-Nearest Neighbors
  4. Hyperparameter Sweep: Cell Grids (4x4, 8x8, 16x16) x Angular Bins (6, 9, 12)
  5. Optical Perturbation Stress Analysis (Exposure, Noise, Alignment, Defocus)
  6. Operational Quality Assurance Gate (ACCEPT/REJECT Logic)
"""

import os
import sys
import json
import time
import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from skimage import exposure
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold

# Output folders
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

TAXONOMY = {
    'crazing': 'Crazing (Thermal Microfissures)',
    'inclusion': 'Inclusion (Foreign Slag Admixture)',
    'patches': 'Patches (Local Layer Delamination)',
    'pitted_surface': 'Pitted Surface (Oxidative Etch Pits)',
    'rolled-in_scale': 'Rolled-in Scale (Scale Entrapment)',
    'scratches': 'Scratches (Frictional Mechanical Gouges)',
    'normal': 'Normal (Defect-Free Reference Plate)'
}

CLASSES = ['normal', 'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
LABEL_INDEX = {c: i for i, c in enumerate(CLASSES)}


def load_industrial_defect_corpus(data_dir="data", target_dim=(128, 128)):
    """Ingest image corpus, perform resolution standardization and contrast normalization."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    if not files:
        raise FileNotFoundError(f"Missing image corpus in {data_dir}")

    images, binary_targets, multiclass_targets, tags, names = [], [], [], [], []
    for f in files:
        bname = os.path.basename(f)
        cat = bname.rsplit('_', 1)[0]
        if cat not in LABEL_INDEX:
            continue

        raw = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue

        norm_img = cv2.resize(raw, target_dim, interpolation=cv2.INTER_AREA)
        images.append(norm_img)
        tags.append(cat)
        names.append(bname)
        multiclass_targets.append(LABEL_INDEX[cat])
        binary_targets.append(0 if cat == 'normal' else 1)

    print(f"[Corpus Ingestion] Total records: {len(images)} | Pristine: {binary_targets.count(0)} | Defective: {binary_targets.count(1)}")
    return np.array(images), np.array(binary_targets), np.array(multiclass_targets), tags, names


def generate_surface_catalog_figure(images, tags):
    """Render catalog montage highlighting distinctive morphological defect patterns."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    fig.patch.set_facecolor('#022c22')  # Dark Emerald/Teal Theme

    class_sample_map = {}
    for i, t in enumerate(tags):
        if t not in class_sample_map:
            class_sample_map[t] = i

    for idx, c in enumerate(CLASSES):
        ax = axes[idx // 4, idx % 4]
        ax.set_facecolor('#064e3b')
        s_idx = class_sample_map[c]
        ax.imshow(images[s_idx], cmap='gray')
        is_norm = (c == 'normal')
        col = '#34d399' if is_norm else '#fb7185'
        status_tag = "GRADE: PASS (NORMAL)" if is_norm else "GRADE: DEFECTIVE"
        ax.set_title(f"{TAXONOMY[c]}\n{status_tag}", color=col, fontsize=10, fontweight='bold', pad=8)
        ax.axis('off')

    # Slot 8: Telemetry panel
    ax_t = axes[1, 3]
    ax_t.set_facecolor('#064e3b')
    ax_t.axis('off')
    info_box = (
        "INSPECTION TAXONOMY\n"
        "-------------------------\n"
        f"Inspected Plates: {len(images)}\n"
        "Geometry: 128x128 Monochromatic\n"
        "Benchmark: NEU Hot-Rolled Strip\n"
        "Baseline Pass Rate: 14.3%\n"
        "Defect Incidence: 85.7%\n"
        "-------------------------\n"
        "Quality Gate: Strict (P >= 0.50)"
    )
    ax_t.text(0.5, 0.5, info_box, color='#a7f3d0', fontsize=9.5, family='monospace',
              ha='center', va='center', bbox=dict(boxstyle='square,pad=0.8', facecolor='#022c22', edgecolor='#059669'))

    plt.suptitle("Figure 1: Industrial Surface Defect Catalog & Microstructure Characterization",
                 fontsize=14, fontweight='bold', color='#ecfdf5', y=0.98)
    plt.tight_layout()
    plt.savefig("figures/surface_defect_taxonomy.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("[Visualization] Generated: figures/surface_defect_taxonomy.png")


def extract_hog_signatures(images, orientations=9, cell_shape=(8, 8), block_shape=(2, 2)):
    """Compute normalized Histogram of Oriented Gradients signatures."""
    features = []
    for img in images:
        vec = hog(img, orientations=orientations, pixels_per_cell=cell_shape,
                  cells_per_block=block_shape, block_norm='L2-Hys', visualize=False)
        features.append(vec)
    return np.array(features)


def visualize_hog_signatures(images, tags):
    """Plot multi-modal gradient signature maps (Raw -> Sobel Magnitude -> HOG Orientation Mesh)."""
    selected = ['normal', 'inclusion', 'patches', 'rolled-in_scale']
    fig, axes = plt.subplots(4, 3, figsize=(13, 15))
    fig.patch.set_facecolor('#022c22')

    for r_idx, c in enumerate(selected):
        img_idx = next(i for i, tag in enumerate(tags) if tag == c)
        sample = images[img_idx]

        _, h_vis = hog(sample, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                       block_norm='L2-Hys', visualize=True)
        h_vis_scaled = exposure.rescale_intensity(h_vis, in_range=(0, 10))

        # Sobel spatial edge energy
        gx = cv2.Sobel(sample, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sample, cv2.CV_32F, 0, 1, ksize=3)
        edge_energy = cv2.magnitude(gx, gy)
        edge_norm = cv2.normalize(edge_energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        axes[r_idx, 0].imshow(sample, cmap='gray')
        axes[r_idx, 0].set_title(f"{TAXONOMY[c]}\n[Input Surface]", color='#ecfdf5', fontsize=10, fontweight='bold')
        axes[r_idx, 0].axis('off')

        axes[r_idx, 1].imshow(edge_norm, cmap='viridis')
        axes[r_idx, 1].set_title(f"{c.upper()}\n[Sobel Edge Flux]", color='#34d399', fontsize=10, fontweight='bold')
        axes[r_idx, 1].axis('off')

        axes[r_idx, 2].imshow(h_vis_scaled, cmap='plasma')
        axes[r_idx, 2].set_title("HOG Spatial Gradient Map\n[8x8 Cells | 9 Bins]", color='#fbbf24', fontsize=10, fontweight='bold')
        axes[r_idx, 2].axis('off')

    plt.suptitle("Figure 2: HOG Directional Descriptor Maps vs Spatial Edge Density",
                 fontsize=14, fontweight='bold', color='#ecfdf5', y=0.99)
    plt.tight_layout()
    plt.savefig("figures/hog_signature_quiver_maps.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("[Visualization] Generated: figures/hog_signature_quiver_maps.png")


def benchmark_defect_classifiers(X, y):
    """Benchmark Linear SVM, Random Forest, and K-Nearest Neighbors across 5-fold CV."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'Linear Support Vector Machine': SVC(kernel='linear', C=1.0, probability=True, random_state=42),
        'Random Forest (100 Trees)': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'K-Nearest Neighbors (k=3)': KNeighborsClassifier(n_neighbors=3)
    }

    metrics_summary = {}
    confusion_dict = {}

    for name, clf in models.items():
        t_start = time.time()
        accs, precs, recs, f1s = [], [], [], []
        pred_acc, true_acc = [], []

        for tr_i, te_i in skf.split(X, y):
            clf.fit(X[tr_i], y[tr_i])
            y_pred = clf.predict(X[te_i])

            accs.append(accuracy_score(y[te_i], y_pred))
            precs.append(precision_score(y[te_i], y_pred, zero_division=0))
            recs.append(recall_score(y[te_i], y_pred, zero_division=0))
            f1s.append(f1_score(y[te_i], y_pred, zero_division=0))

            pred_acc.extend(y_pred)
            true_acc.extend(y[te_i])

        dt = time.time() - t_start
        cm = confusion_matrix(true_acc, pred_acc)

        metrics_summary[name] = {
            'accuracy': float(np.mean(accs)),
            'precision': float(np.mean(precs)),
            'recall': float(np.mean(recs)),
            'f1_score': float(np.mean(f1s)),
            'evaluation_latency_ms': float((dt / len(X)) * 1000)
        }
        confusion_dict[name] = cm
        print(f"[Model Benchmark] {name:<32} | Acc: {np.mean(accs)*100:.1f}% | F1: {np.mean(f1s)*100:.1f}% | Latency: {(dt/len(X))*1000:.2f}ms")

    # Plot confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#022c22')
    palettes = ['Greens', 'YlGnBu', 'BuPu']

    for idx, (m_name, cm) in enumerate(confusion_dict.items()):
        ax = axes[idx]
        ax.set_facecolor('#064e3b')
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap=palettes[idx],
                    xticklabels=['Pass', 'Defect'], yticklabels=['Pass', 'Defect'],
                    cbar=False, ax=ax, annot_kws={"size": 12, "weight": "bold"})
        ax.set_title(f"{m_name}\nAcc: {metrics_summary[m_name]['accuracy']*100:.1f}% | F1: {metrics_summary[m_name]['f1_score']*100:.1f}%",
                     color='#ecfdf5', fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Predicted Inspection Decision", color='#a7f3d0', fontsize=10)
        ax.set_ylabel("True Ground Truth", color='#a7f3d0', fontsize=10)
        ax.tick_params(colors='#a7f3d0')

    plt.suptitle("Figure 3: Classifier Benchmark Confusion Matrices (Normalized)",
                 fontsize=14, fontweight='bold', color='#ecfdf5', y=1.02)
    plt.tight_layout()
    plt.savefig("figures/confusion_matrices_comparison.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    with open("results/metrics_benchmark.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print("[Benchmarking] Saved: results/metrics_benchmark.json")
    return metrics_summary


def run_hog_hyperparameter_sweep(images, y):
    """Grid sweep over cell geometries (4x4, 8x8, 16x16) and orientation bins (6, 9, 12)."""
    cell_grids = [(4, 4), (8, 8), (16, 16)]
    bin_counts = [6, 9, 12]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    sweep_records = []
    print("\n[Hyperparameter Sweep] Evaluating 9 HOG Configurations:")
    for cell in cell_grids:
        for bins in bin_counts:
            t0 = time.time()
            feats = []
            for img in images:
                f = hog(img, orientations=bins, pixels_per_cell=cell, cells_per_block=(2, 2),
                        block_norm='L2-Hys', visualize=False)
                feats.append(f)
            X_arr = np.array(feats)
            t_extract = (time.time() - t0) / len(images)

            clf = SVC(kernel='linear', C=1.0, random_state=42)
            accs, f1s = [], []
            for tr_i, te_i in skf.split(X_arr, y):
                clf.fit(X_arr[tr_i], y[tr_i])
                preds = clf.predict(X_arr[te_i])
                accs.append(accuracy_score(y[te_i], preds))
                f1s.append(f1_score(y[te_i], preds, zero_division=0))

            rec = {
                'cell_size': f"{cell[0]}x{cell[1]}",
                'bins': bins,
                'feature_dimension': int(X_arr.shape[1]),
                'extraction_latency_ms': float(t_extract * 1000),
                'accuracy': float(np.mean(accs)),
                'f1_score': float(np.mean(f1s))
            }
            sweep_records.append(rec)
            print(f"  Cell: {rec['cell_size']:>5} | Bins: {bins:>2} | Dim: {rec['feature_dimension']:>5} | Time: {rec['extraction_latency_ms']:.2f}ms | Acc: {rec['accuracy']*100:.1f}% | F1: {rec['f1_score']*100:.1f}%")

    with open("results/parameter_grid_search.json", "w") as f:
        json.dump(sweep_records, f, indent=4)

    # Plot ablation charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.patch.set_facecolor('#022c22')

    # Cell size analysis at fixed 9 bins
    fixed_bins_9 = [r for r in sweep_records if r['bins'] == 9]
    c_labels = [r['cell_size'] for r in fixed_bins_9]
    c_accs = [r['accuracy'] * 100 for r in fixed_bins_9]
    c_f1s = [r['f1_score'] * 100 for r in fixed_bins_9]
    dims = [r['feature_dimension'] for r in fixed_bins_9]

    idx = np.arange(len(c_labels))
    w = 0.35
    ax1.set_facecolor('#064e3b')
    ax1.bar(idx - w/2, c_accs, w, label='Accuracy (%)', color='#34d399', edgecolor='#059669')
    ax1.bar(idx + w/2, c_f1s, w, label='F1-Score (%)', color='#fbbf24', edgecolor='#d97706')
    ax1.set_xticks(idx)
    ax1.set_xticklabels([f"{l}\n(Dim: {d})" for l, d in zip(c_labels, dims)], color='#a7f3d0')
    ax1.set_title("Cell Dimension Sensitivity (Fixed 9 Bins)", color='#ecfdf5', fontsize=11, fontweight='bold')
    ax1.set_ylabel("Metric (%)", color='#a7f3d0')
    ax1.set_ylim(70, 105)
    ax1.legend(facecolor='#022c22', edgecolor='#059669', labelcolor='#ecfdf5')
    ax1.grid(color='#047857', linestyle=':', alpha=0.6)

    # Bin count analysis at fixed 8x8 cell
    fixed_cell_8 = [r for r in sweep_records if r['cell_size'] == '8x8']
    b_labels = [f"{r['bins']} Bins" for r in fixed_cell_8]
    b_accs = [r['accuracy'] * 100 for r in fixed_cell_8]
    b_f1s = [r['f1_score'] * 100 for r in fixed_cell_8]
    times = [r['extraction_latency_ms'] for r in fixed_cell_8]

    ax2.set_facecolor('#064e3b')
    ax2.bar(idx - w/2, b_accs, w, label='Accuracy (%)', color='#2dd4bf', edgecolor='#0f766e')
    ax2.bar(idx + w/2, b_f1s, w, label='F1-Score (%)', color='#f472b6', edgecolor='#db2777')
    ax2.set_xticks(idx)
    ax2.set_xticklabels([f"{l}\n({t:.1f}ms)" for l, t in zip(b_labels, times)], color='#a7f3d0')
    ax2.set_title("Angular Quantization Sensitivity (Fixed 8x8 Cell)", color='#ecfdf5', fontsize=11, fontweight='bold')
    ax2.set_ylabel("Metric (%)", color='#a7f3d0')
    ax2.set_ylim(70, 105)
    ax2.legend(facecolor='#022c22', edgecolor='#059669', labelcolor='#ecfdf5')
    ax2.grid(color='#047857', linestyle=':', alpha=0.6)

    plt.suptitle("Figure 4: Hyperparameter Sweep - Feature Dimension vs Precision Response",
                 fontsize=14, fontweight='bold', color='#ecfdf5', y=1.02)
    plt.tight_layout()
    plt.savefig("figures/ablation_parameter_curves.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    return sweep_records


def evaluate_imaging_stress_tests(images, y):
    """Stress test classifier performance under exposure changes, sensor noise, rotation, and defocus."""
    np.random.seed(42)
    X_clean = extract_hog_signatures(images)
    clf = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    clf.fit(X_clean, y)

    base_preds = clf.predict(X_clean)
    b_acc = accuracy_score(y, base_preds)
    b_f1 = f1_score(y, base_preds)

    def perturb(img, mode):
        h, w = img.shape
        if mode == 'underexposure':
            return np.clip(img.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)
        elif mode == 'overexposure':
            return np.clip(img.astype(np.float32) * 1.45, 0, 255).astype(np.uint8)
        elif mode == 'gaussian_noise':
            n = np.random.normal(0, 22, img.shape)
            return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
        elif mode == 'skew_rotation':
            M = cv2.getRotationMatrix2D((w // 2, h // 2), -12, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif mode == 'defocus_blur':
            return cv2.GaussianBlur(img, (9, 9), 3.0)
        return img

    stress_regimes = [
        ('Nominal Reference', 'clean'),
        ('Underexposure (0.55x)', 'underexposure'),
        ('Overexposure (1.45x)', 'overexposure'),
        ('Additive Noise (sigma=22)', 'gaussian_noise'),
        ('Conveyor Skew (-12 deg)', 'skew_rotation'),
        ('Defocus Blur (sigma=3.0)', 'defocus_blur')
    ]

    stress_output = []
    sample_montage = []

    print("\n[Stress Testing] Evaluating System Robustness:")
    for label, mode in stress_regimes:
        if mode == 'clean':
            stress_output.append({
                'scenario': label,
                'accuracy': float(b_acc),
                'f1_score': float(b_f1),
                'accuracy_loss': 0.0,
                'f1_loss': 0.0
            })
            sample_montage.append(images[1])
            continue

        p_imgs = [perturb(im, mode) for im in images]
        sample_montage.append(p_imgs[1])
        X_p = extract_hog_signatures(p_imgs)
        preds = clf.predict(X_p)

        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds)

        stress_output.append({
            'scenario': label,
            'accuracy': float(acc),
            'f1_score': float(f1),
            'accuracy_loss': float(b_acc - acc),
            'f1_loss': float(b_f1 - f1)
        })
        print(f"  Regime: {label:<26} | Acc: {acc*100:.1f}% (Drop: -{(b_acc-acc)*100:.1f}%) | F1: {f1*100:.1f}%")

    with open("results/imaging_perturbation_results.json", "w") as f:
        json.dump(stress_output, f, indent=4)

    # Plot degradation charts
    fig = plt.figure(figsize=(16, 9.5))
    fig.patch.set_facecolor('#022c22')
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.25])

    for i, ((name, _), smp) in enumerate(zip(stress_regimes, sample_montage)):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor('#064e3b')
        ax.imshow(smp, cmap='gray')
        ax.set_title(name.split(' (')[0], color='#a7f3d0', fontsize=9.5, fontweight='bold', pad=6)
        ax.axis('off')

    ax_bar = fig.add_subplot(gs[1, :])
    ax_bar.set_facecolor('#064e3b')

    names = [r['scenario'] for r in stress_output]
    accs = [r['accuracy'] * 100 for r in stress_output]
    f1s = [r['f1_score'] * 100 for r in stress_output]
    idx = np.arange(len(names))
    w = 0.35

    ax_bar.bar(idx - w/2, accs, w, label='Accuracy (%)', color='#34d399', edgecolor='#059669')
    ax_bar.bar(idx + w/2, f1s, w, label='F1-Score (%)', color='#fb7185', edgecolor='#e11d48')
    ax_bar.set_xticks(idx)
    ax_bar.set_xticklabels(names, rotation=15, ha='right', color='#a7f3d0', fontsize=9.5)
    ax_bar.set_ylabel('Performance (%)', color='#a7f3d0', fontsize=11)
    ax_bar.set_ylim(40, 105)
    ax_bar.legend(facecolor='#022c22', edgecolor='#059669', labelcolor='#ecfdf5', loc='lower left')
    ax_bar.grid(color='#047857', linestyle=':', alpha=0.6)

    for i in range(1, len(stress_output)):
        loss = stress_output[i]['accuracy_loss'] * 100
        ax_bar.annotate(f"-{loss:.1f}%", xy=(idx[i] - w/2, accs[i]), xytext=(0, 4),
                        textcoords="offset points", ha='center', color='#fbbf24', fontsize=9, fontweight='bold')

    plt.suptitle("Figure 5: Environmental Perturbation Stress Profile (Degradation Response)",
                 fontsize=14, fontweight='bold', color='#ecfdf5', y=0.98)
    plt.tight_layout()
    plt.savefig("figures/imaging_degradation_profile.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    return stress_output


def execute_industrial_qc_inspection(clf, test_image_path, threshold=0.50):
    """Factory Quality Control Module executing final automated Accept/Reject verdicts."""
    raw = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"Cannot load image {test_image_path}")

    res = cv2.resize(raw, (128, 128), interpolation=cv2.INTER_AREA)
    feat = hog(res, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
               block_norm='L2-Hys').reshape(1, -1)

    probabilities = clf.predict_proba(feat)[0]
    p_defect = probabilities[1]
    is_rejected = p_defect >= threshold
    conf = p_defect if is_rejected else probabilities[0]

    decision = {
        'prediction': "DEFECTIVE" if is_rejected else "NON-DEFECTIVE",
        'confidence_pct': round(float(conf * 100), 2),
        'action': "REJECT PRODUCT" if is_rejected else "ACCEPT PRODUCT"
    }

    print("\n========================================")
    print("PRODUCT INSPECTION RESULT")
    print("========================================")
    print(f"Prediction: {decision['prediction']}")
    print(f"Confidence: {decision['confidence_pct']}%")
    print(f"Action: {decision['action']}")
    print("========================================")
    return decision, res


def generate_qc_report_cards(clf, images, tags, names):
    """Render pass/reject visual report cards."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    fig.patch.set_facecolor('#022c22')

    for i in range(8):
        ax = axes[i // 4, i % 4]
        ax.set_facecolor('#064e3b')
        img = images[i]
        feat = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                   block_norm='L2-Hys').reshape(1, -1)
        prob = clf.predict_proba(feat)[0]
        is_def = prob[1] >= 0.50
        conf = prob[1] if is_def else prob[0]
        status_txt = "REJECT PRODUCT" if is_def else "ACCEPT PRODUCT"
        pred_txt = "DEFECTIVE" if is_def else "NON-DEFECTIVE"
        border_col = '#fb7185' if is_def else '#34d399'

        ax.imshow(img, cmap='gray')
        ax.set_title(f"{status_txt}\nPred: {pred_txt} ({conf*100:.1f}%)",
                     color=border_col, fontsize=9.5, fontweight='bold', pad=6)
        ax.axis('off')

        rect = plt.Rectangle((0, 0), img.shape[1], img.shape[0],
                             linewidth=3, edgecolor=border_col, facecolor='none')
        ax.add_patch(rect)

    plt.suptitle("Figure 6: Automated QC Inspection Telemetry & Pass/Reject Output",
                 fontsize=14, fontweight='bold', color='#ecfdf5', y=0.98)
    plt.tight_layout()
    plt.savefig("figures/quality_gate_telemetry.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("[Visualization] Generated: figures/quality_gate_telemetry.png")


def main():
    print("======================================================================")
    print("STARTING LAB 05 DEFECT DETECTION PIPELINE | AFNAN KHAN (@Afnan-makstrix)")
    print("======================================================================")

    # 1. Ingest Data
    images, y_bin, y_mul, tags, names = load_industrial_defect_corpus()
    generate_surface_catalog_figure(images, tags)

    # 2. Extract Signatures
    X_hog = extract_hog_signatures(images)
    visualize_hog_signatures(images, tags)

    # 3. Benchmark Classifiers
    metrics = benchmark_defect_classifiers(X_hog, y_bin)

    # 4. Hyperparameter Sweep
    sweep = run_hog_hyperparameter_sweep(images, y_bin)

    # 5. Stress Testing
    stress = evaluate_imaging_stress_tests(images, y_bin)

    # 6. Quality Control Gate
    final_clf = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    final_clf.fit(X_hog, y_bin)

    print("\n[Test Inspection 1 - Pristine Reference]")
    execute_industrial_qc_inspection(final_clf, os.path.join("data", "normal_1.jpg"))

    print("\n[Test Inspection 2 - Inclusion Flaw]")
    execute_industrial_qc_inspection(final_clf, os.path.join("data", "inclusion_1.jpg"))

    generate_qc_report_cards(final_clf, images, tags, names)
    print("\n[Status] Complete Lab 05 pipeline executed successfully for Afnan Khan!")

if __name__ == '__main__':
    main()
