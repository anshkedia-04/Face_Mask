# import os
# import pickle
# import numpy as np
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
# from PIL import Image, ImageEnhance
# import torch
# from tqdm import tqdm
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# # ==============================
# # Configuration
# # ==============================
# DATASET_DIR = 'dataset'
# EMBEDDINGS_FILE = "student_embeddings.pkl"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Initialize FaceNet
# mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)  # margin helps capture full face
# resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# # ==============================
# # Helper: Data Augmentation
# # ==============================
# def augment_image(img):
#     """Apply simple augmentations to increase training samples."""
#     augmented = []
#     # Brightness
#     enhancer = ImageEnhance.Brightness(img)
#     augmented.append(enhancer.enhance(1.2))
#     augmented.append(enhancer.enhance(0.8))
#     # Contrast
#     enhancer = ImageEnhance.Contrast(img)
#     augmented.append(enhancer.enhance(1.3))
#     # Flips
#     augmented.append(img.transpose(Image.FLIP_LEFT_RIGHT))
#     return augmented

# # ==============================
# # Load existing embeddings
# # ==============================
# if os.path.exists(EMBEDDINGS_FILE):
#     print(f"📂 Found {EMBEDDINGS_FILE}, loading existing embeddings...")
#     with open(EMBEDDINGS_FILE, "rb") as f:
#         data = pickle.load(f)
#     X, y = list(data["X"]), list(data["y"])
# else:
#     print("🆕 No embeddings found. Creating new file...")
#     X, y = [], []

# # ==============================
# # Process Dataset
# # ==============================
# for folder in tqdm(os.listdir(DATASET_DIR), desc="Processing dataset"):
#     person_path = os.path.join(DATASET_DIR, folder)
#     if not os.path.isdir(person_path):
#         continue

#     current_images = [img for img in os.listdir(person_path) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]

#     for img_name in current_images:
#         img_path = os.path.join(person_path, img_name)
#         try:
#             img = Image.open(img_path).convert('RGB')
#         except:
#             continue

#         # Original embedding
#         face = mtcnn(img)
#         if face is not None:
#             with torch.no_grad():
#                 emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()[0]
#                 X.append(emb)
#                 y.append(folder)

#         # Augmented embeddings
#         for aug_img in augment_image(img):
#             face_aug = mtcnn(aug_img)
#             if face_aug is not None:
#                 with torch.no_grad():
#                     emb_aug = resnet(face_aug.unsqueeze(0).to(device)).cpu().numpy()[0]
#                     X.append(emb_aug)
#                     y.append(folder)

# # ==============================
# # Save embeddings
# # ==============================
# with open(EMBEDDINGS_FILE, "wb") as f:
#     pickle.dump({"X": np.array(X), "y": np.array(y)}, f)
# print(f"✅ Embeddings saved to {EMBEDDINGS_FILE}")

# # ==============================
# # Train-Test Split & SVM
# # ==============================
# X = np.array(X)
# y = np.array(y)

# # Normalize embeddings (important for SVM performance)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Encode labels
# encoder = LabelEncoder()
# y_encoded = encoder.fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
# )

# clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)  # RBF works better for embeddings
# clf.fit(X_train, y_train)

# train_acc = accuracy_score(y_train, clf.predict(X_train))
# test_acc = accuracy_score(y_test, clf.predict(X_test))

# print(f"\n✅ Final Training Accuracy: {train_acc * 100:.2f}%")
# print(f"✅ Final Validation/Test Accuracy: {test_acc * 100:.2f}%")
# print("\n📊 Classification Report:\n", classification_report(y_test, clf.predict(X_test), target_names=encoder.classes_))


import os
import pickle
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image
import torch
from tqdm import tqdm

# Configuration
DATASET_DIR = 'dataset'
EMBEDDINGS_FILE = "student_embeddings.pkl"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize FaceNet
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ==============================
# Load existing embeddings (if any)
# ==============================
if os.path.exists(EMBEDDINGS_FILE):
    print(f"📂 Found {EMBEDDINGS_FILE}, loading existing embeddings...")
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    X, y, student_embeddings = list(data["X"]), list(data["y"]), dict(data["student_embeddings"])
else:
    print("🆕 No embeddings found. Creating new file...")
    X, y, student_embeddings = [], [], {}

# ==============================
# Check for new images
# ==============================
new_added = False
for folder in tqdm(os.listdir(DATASET_DIR), desc="Processing dataset"):
    person_path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(person_path):
        continue

    # If student already exists in embeddings, skip unless new images are added
    existing_count = sum(1 for name in y if name == folder)
    current_images = [img for img in os.listdir(person_path) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if existing_count >= len(current_images):
        continue  # No new images for this student

    print(f"➕ Found new images for {folder}, generating embeddings...")
    person_embeddings = []

    for img_name in current_images:
        img_path = os.path.join(person_path, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            continue

        face = mtcnn(img)
        if face is not None:
            with torch.no_grad():
                emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()[0]
                X.append(emb)
                y.append(folder)
                person_embeddings.append(emb)

    if person_embeddings:
        student_embeddings[folder] = np.mean(person_embeddings, axis=0)
        new_added = True

# ==============================
# Save updated embeddings
# ==============================
if new_added:
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump({"X": np.array(X), "y": np.array(y), "student_embeddings": student_embeddings}, f)
    print(f"✅ Updated embeddings saved to {EMBEDDINGS_FILE}")
else:
    print("ℹ️ No new data found. Using existing embeddings.")

# ==============================
# Train-Test Split & SVM
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc = accuracy_score(y_test, clf.predict(X_test))

print(f"\n✅ Final Training Accuracy: {train_acc * 100:.2f}%")
print(f"✅ Final Validation/Test Accuracy: {test_acc * 100:.2f}%")
