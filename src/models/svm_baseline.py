"""Classical baseline: HOG + colour statistics -> StandardScaler -> RBF SVM."""
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def extract_features(img_uint8, hog_size=64):
    small = np.asarray(Image.fromarray(img_uint8).resize((hog_size, hog_size), Image.BILINEAR))
    gray = small.mean(axis=2)
    h = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    px = img_uint8.reshape(-1, 3).astype(np.float32)
    hist = np.concatenate([np.histogram(px[:, c], bins=8, range=(0, 255), density=True)[0] for c in range(3)])
    return np.concatenate([h, hist, px.mean(0), px.std(0)]).astype(np.float32)


class SVMBaseline:
    def __init__(self, C=10.0, gamma="scale"):
        self.model = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, kernel="rbf",
                                                         class_weight="balanced", decision_function_shape="ovr"))

    def features(self, images):
        return np.stack([extract_features(np.asarray(im)) for im in images])

    def fit(self, images, y):
        self.model.fit(self.features(images), y)
        return self

    def probs(self, images):
        """Softmax over the one-vs-rest decision scores (the SVM is not natively probabilistic)."""
        s = self.model.decision_function(self.features(images))
        s = s - s.max(1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(1, keepdims=True)
