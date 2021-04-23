import random
import cv2
from matplotlib import pyplot as plt
import lbp
import lbph
from definitions import *

TRAINING_SET_LEN = 135

train_files = []
validation_files = []
subjects = set()
train_data = {}
validation_data = {}
calculated_data = {}

img1 = cv2.imread(SINGLE_IMG)
gray_img_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread(os.path.join(PROJECT_ROOT, 'Resources/single_face_2.jpg'))
gray_img_2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img3 = cv2.imread(os.path.join(PROJECT_ROOT, 'Resources/single_face_3.jpg'))
gray_img_3 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(FACE_PATTERN)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)
    (x, y, w, h) = faces[0]
    return gray[y:y + w, x:x + h], faces[0]


def load_images_from_folder(folder):
    file_selected = [f for f in os.listdir(folder)]
    file_selected = random.sample(file_selected, TRAINING_SET_LEN)
    for filename in file_selected:
        if filename.__contains__('.jpg'):
            train_files.append(filename)
            subject = filename.split('.')[0]
            img = cv2.imread(os.path.join(folder, filename))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if gray is not None:
                train_data[subject].append(gray)


def init_train_data(folder):
    subjects = names = list(set([f.split('.')[0] for f in os.listdir(folder)]))
    for filename in names:
        images = []
        train_data[filename] = images


def init_validation_data(folder):
    names = set([f for f in os.listdir(folder)])
    t_files = set(train_files)
    v_names = names - t_files
    for filename in v_names:
        data = []
        if filename.__contains__('.jpg'):
            img = cv2.imread(os.path.join(folder, filename))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if gray is not None:
                data.append(lbph.hist(gray))
                validation_data[filename] = data


def train_model():
    for k, v in train_data.items():
        if len(v) > 0:
            calculated_data[k] = lbph.class_hist(v)


def predict_all():
    for k, v in validation_data.items():
        similar_hist = lbph.find_closest_histogram(calculated_data, v[0], ABSOLUTE)
        v.append(similar_hist)


def show_predicted_data(folder):
    for file in validation_data.keys():
        img = cv2.imread(os.path.join(folder, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        haar_face_cascade = cv2.CascadeClassifier(FACE_PATTERN)
        faces = haar_face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, validation_data[file][1], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        plt.imshow(img)
        plt.show()


def show_class_histograms():
    for k, v in calculated_data.items():
        plt.title(k)
        plt.hist(v)
        plt.show()


def metrics():
    errores = 0
    aciertos = 0
    for k, v in validation_data.items():
        if v[1] in k:
            aciertos = aciertos + 1
        else:
            errores = errores + 1
    pres = aciertos / (aciertos + errores)
    pres = pres * 100
    acc = (aciertos + errores) / TRAINING_SET_LEN
    acc = acc * 100
    print("Precision: {:.2f}".format(pres))
    print("Accuracy: {:.2f}".format(acc))


def show_example_lbp():
    ex_img = cv2.imread(SINGLE_IMG)
    g_img = cv2.cvtColor(ex_img, cv2.COLOR_BGR2GRAY)
    g_img = cv2.resize(g_img, (512, 512))
    plt.imshow(g_img)
    plt.show()
    limg = lbp.calculate_lbp(g_img)
    plt.imshow(limg)
    plt.show()


init_train_data(FACES_DIR)
load_images_from_folder(FACES_DIR)
init_validation_data(FACES_DIR)
train_model()
show_class_histograms()
predict_all()
show_predicted_data(FACES_DIR)
metrics()
#show_example_lbp()

