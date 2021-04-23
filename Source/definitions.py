import os
from utils import get_project_root, get_source_root

ROOT_DIR = get_source_root()
PROJECT_ROOT = get_project_root()
SINGLE_IMG = os.path.join(PROJECT_ROOT, 'Resources/single_face.jpg')
FACES_DIR = os.path.join(PROJECT_ROOT, 'Resources/yalefaces')
FACE_PATTERN = os.path.join(ROOT_DIR, 'haarcascade_frontalface_alt.xml')
# CONFIG_PATH = os.path.join(ROOT_DIR, 'configuration.conf')

CHI_SQUARE = "ChiSquare"
EUCLIDEAN = "Euclidean"
NORMALIZED_EUCLIDEAN = "NormalizedEuclidean"
ABSOLUTE = "Absolute"
