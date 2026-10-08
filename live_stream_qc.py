"""
Live Stream Quality Gate Prototype
Student: Afnan Khan (@Afnan-makstrix)
Course: Computer Vision
"""

import cv2
import time
import glob
import os
import numpy as np
from skimage.feature import hog
from skimage import exposure
from sklearn.svm import SVC

def train_qc_engine():
    data_files = sorted(glob.glob("data/*.jpg"))
    if not data_files:
        return None
    X, y = [], []
    for f in data_files:
        c = os.path.basename(f).rsplit('_', 1)[0]
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        res = cv2.resize(im, (128, 128))
        feat = hog(res, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        X.append(feat)
        y.append(0 if c == 'normal' else 1)
    clf = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    clf.fit(X, y)
    return clf

def main():
    clf = train_qc_engine()
    if clf is None:
        print("Training data missing.")
        return

    cap = cv2.VideoCapture(0)
    has_cam = cap.isOpened()
    stream_files = sorted(glob.glob("data/*.jpg"))
    ptr = 0

    print("Starting Live Inspection Stream [Afnan Khan Prototype]...")
    print("Press 'q' to exit stream.")

    while True:
        if has_cam:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            target = cv2.resize(gray, (128, 128))
            view = frame.copy()
        else:
            time.sleep(0.09)
            raw = cv2.imread(stream_files[ptr % len(stream_files)])
            ptr += 1
            view = cv2.resize(raw, (640, 480))
            gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
            target = cv2.resize(gray, (128, 128))

        feat, vis = hog(target, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                        block_norm='L2-Hys', visualize=True)
        prob = clf.predict_proba(feat.reshape(1, -1))[0]
        is_def = prob[1] >= 0.50
        conf = prob[1] if is_def else prob[0]

        # Draw HUD banner
        h, w, _ = view.shape
        hud_col = (50, 50, 230) if is_def else (60, 200, 60)
        hud_txt = "ACTION: REJECT [DEFECTIVE]" if is_def else "ACTION: ACCEPT [PASS]"

        cv2.rectangle(view, (0, 0), (w, 55), (15, 25, 20), -1)
        cv2.putText(view, hud_txt, (18, 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, hud_col, 2)
        cv2.putText(view, f"Confidence: {conf*100:.1f}%", (w - 240, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 240, 230), 1)

        # Inset HOG gradient map
        vis_u8 = exposure.rescale_intensity(vis, in_range=(0, 10), out_range=(0, 255)).astype(np.uint8)
        color_hog = cv2.applyColorMap(vis_u8, cv2.COLORMAP_VIRIDIS)
        view[h - 130:h - 10, w - 130:w - 10] = cv2.resize(color_hog, (120, 120))
        cv2.rectangle(view, (w - 130, h - 130), (w - 10, h - 10), (255, 255, 255), 1)

        cv2.imshow("Industrial Automated Defect Gate - Afnan Khan", view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if has_cam:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
