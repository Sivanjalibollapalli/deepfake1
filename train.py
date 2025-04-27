import os
import json
import shutil
import random
import argparse
import cv2
from PIL import Image
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def create_split_subdirs(split_dir):
    real_dir = os.path.join(split_dir, 'real')
    fake_dir = os.path.join(split_dir, 'fake')
    real_frame_dir = os.path.join(split_dir, 'real_frame')
    fake_frame_dir = os.path.join(split_dir, 'fake_frame')

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    os.makedirs(real_frame_dir, exist_ok=True)
    os.makedirs(fake_frame_dir, exist_ok=True)

    return real_dir, fake_dir, real_frame_dir, fake_frame_dir

def split_videos(metadata_path, input_dir, sample_size=10, val_size=0.3, test_size_relative=0.5, random_seed=42):
    random.seed(random_seed)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    real_videos, fake_videos = [], []

    for video_name, data in metadata.items():
        video_path = os.path.join(input_dir, video_name)
        if os.path.exists(video_path):
            if data['label'] == 'REAL':
                real_videos.append((video_path, 'real'))
            elif data['label'] == 'FAKE':
                fake_videos.append((video_path, 'fake'))

    real_videos = random.sample(real_videos, min(len(real_videos), sample_size))
    fake_videos = random.sample(fake_videos, min(len(fake_videos), sample_size))

    all_videos = real_videos + fake_videos
    labels = [1 if v[1] == 'real' else 0 for v in all_videos]

    train_videos, temp_videos, train_labels, temp_labels = train_test_split(
        all_videos, labels, test_size=val_size + test_size_relative * val_size, stratify=labels, random_state=random_seed)
    val_videos, test_videos, val_labels, test_labels = train_test_split(
        temp_videos, temp_labels, test_size=test_size_relative / (1 + test_size_relative), stratify=temp_labels, random_state=random_seed)

    return train_videos, val_videos, test_videos

def copy_videos_to_split_dirs(videos, split_dir):
    real_dir, fake_dir, _, _ = create_split_subdirs(split_dir)

    for video_path, label in videos:
        target_dir = real_dir if label == 'real' else fake_dir
        video_name = os.path.basename(video_path)
        shutil.copy(video_path, os.path.join(target_dir, video_name))
        print(f"Copied {video_name} to {target_dir}")

def extract_faces_from_video(video_path, output_dir, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    face_count = 0
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

        for i, (x, y, w, h) in enumerate(faces):
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (224, 224))
            output_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_frame{frame_count}_face{i}.jpg")
            cv2.imwrite(output_path, face_resized)
            face_count += 1
    cap.release()
    print(f"{video_path}: {face_count} faces extracted.")

def extract_faces_from_videos(split_dir, frame_skip=10):
    real_dir, fake_dir, real_frame_dir, fake_frame_dir = create_split_subdirs(split_dir)

    print(f"Extracting faces from real videos in {split_dir}...")
    for video_name in os.listdir(real_dir):
        video_path = os.path.join(real_dir, video_name)
        if os.path.isfile(video_path):
            extract_faces_from_video(video_path, real_frame_dir, frame_skip)

    print(f"Extracting faces from fake videos in {split_dir}...")
    for video_name in os.listdir(fake_dir):
        video_path = os.path.join(fake_dir, video_name)
        if os.path.isfile(video_path):
            extract_faces_from_video(video_path, fake_frame_dir, frame_skip)

def normalize_images(input_dir, output_dir, image_size=224):
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(input_dir):
        try:
            img_path = os.path.join(input_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((image_size, image_size))
            img.save(os.path.join(output_dir, img_name))
        except Exception as e:
            print(f"Error normalizing image {img_name}: {e}")

class FaceDataset(Dataset):
    def __init__(self, norm_dir, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

        real_dir = os.path.join(norm_dir, 'real')
        fake_dir = os.path.join(norm_dir, 'fake')

        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                img_path = os.path.join(real_dir, img_name)
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append(img_path)
                    self.labels.append(0)

        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                img_path = os.path.join(fake_dir, img_name)
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append(img_path)
                    self.labels.append(1)

        if not self.data:
            print(f"Warning: No data found in {norm_dir}!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            img = Image.new('RGB', (224, 224))

        if self.transform:
            img = self.transform(img)

        return img, label

def evaluate_model(model, loader, criterion, device):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, early_stopping, history):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} | "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    return history

def test_model(model, test_loader, criterion, device):
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return test_loss, test_accuracy

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path, model, device):
    model.load_state_dict(torch.load(file_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {file_path}")
    return model

def plot_history(history):
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_accuracy = [h['train_accuracy'] for h in history]
    val_accuracy = [h['val_accuracy'] for h in history]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label="Training Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def evaluate_classification_metrics(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_true, y_pred):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ViT for Deepfake Detection')
    parser.add_argument('--metadata_path', type=str, default='/kaggle/input/dfdc-48/dfdc_train_part_48/metadata.json', help='Path to metadata file')
    parser.add_argument('--input_dir', type=str, default='/kaggle/input/dfdc-48/dfdc_train_part_48', help='Path to input video directory')
    parser.add_argument('--base_dir', type=str, default='/kaggle/working/dataset', help='Base directory for dataset splits')
    parser.add_argument('--sample_size', type=int, default=20, help='Number of videos to sample from each class')
    parser.add_argument('--frame_skip', type=int, default=10, help='Frame skip rate for face extraction')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--patience', type=int, default=40, help='Patience for early stopping')
    parser.add_argument('--output_model_path', type=str, default='/kaggle/working/vit_deepfake.pth', help='Path to save trained model')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    train_dir = os.path.join(args.base_dir, 'train')
    val_dir = os.path.join(args.base_dir, 'validation')
    test_dir = os.path.join(args.base_dir, 'test')
    norm_train_dir = 'norm_train_dir'
    norm_val_dir = 'norm_val_dir'
    norm_test_dir = 'norm_test_dir'

    for split_dir in [train_dir, val_dir, test_dir]:
        os.makedirs(split_dir, exist_ok=True)
        create_split_subdirs(split_dir)

    train_videos, val_videos, test_videos = split_videos(args.metadata_path, args.input_dir, sample_size=args.sample_size, random_seed=args.random_seed)
    copy_videos_to_split_dirs(train_videos, train_dir)
    copy_videos_to_split_dirs(val_videos, val_dir)
    copy_videos_to_split_dirs(test_videos, test_dir)

    extract_faces_from_videos(train_dir, frame_skip=args.frame_skip)
    extract_faces_from_videos(val_dir, frame_skip=args.frame_skip)
    extract_faces_from_videos(test_dir, frame_skip=args.frame_skip)

    normalize_images(os.path.join(train_dir, "fake_frame"), os.path.join(norm_train_dir, "fake"))
    normalize_images(os.path.join(train_dir, "real_frame"), os.path.join(norm_train_dir, "real"))
    normalize_images(os.path.join(test_dir, "fake_frame"), os.path.join(norm_test_dir, "fake"))
    normalize_images(os.path.join(test_dir, "real_frame"), os.path.join(norm_test_dir, "real"))
    normalize_images(os.path.join(val_dir, "fake_frame"), os.path.join(norm_val_dir, "fake"))
    normalize_images(os.path.join(val_dir, "real_frame"), os.path.join(norm_val_dir, "real"))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = FaceDataset(norm_train_dir, transform=transform)
    val_dataset = FaceDataset(norm_val_dir, transform=transform)
    test_dataset = FaceDataset(norm_test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
    vit_model.head = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features=768, out_features=2)
    )
    vit_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vit_model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    history = []

    print("Starting training...")
    history = train_model(vit_model, train_loader, val_loader, criterion, optimizer, scheduler, args.num_epochs, device, early_stopping, history)
    print("Training finished!")

    test_loss, test_accuracy = test_model(vit_model, test_loader, criterion, device)
    save_model(vit_model, args.output_model_path)

    print("\nPerformance Evaluation on Test Set:")
    evaluate_classification_metrics(vit_model, test_loader, device)

    plot_history(history)