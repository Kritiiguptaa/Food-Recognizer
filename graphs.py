import os
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import torch.nn.functional as F

# --- CONFIGURATION ---
ROOT = Path(__file__).resolve().parent
CUISINE_ROOT = ROOT / "cuisines"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"⚙️  Running on device: {device}")

# Define Transforms (Must match training)
TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# --- HELPER FUNCTIONS ---

def load_model(path, num_classes):
    """Loads EfficientNet B0 or B3 based on filename."""
    path = str(path)
    if "b3" in path.lower():
        print(f"   Model detected: EfficientNet B3")
        from torchvision.models import efficientnet_b3
        m = efficientnet_b3(weights=None) 
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        print(f"   Model detected: EfficientNet B0")
        from torchvision.models import efficientnet_b0
        m = efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    try:
        state = torch.load(path, map_location=device)
        m.load_state_dict(state)
        m.to(device).eval()
        return m
    except Exception as e:
        print(f"❌ Error loading model at {path}: {e}")
        return None

def plot_top_k_bar(top1_acc, top3_acc, name, save_name):
    """Plots a bar chart comparing Top-1 vs Top-3 accuracy."""
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Top-1 Accuracy', 'Top-3 Accuracy'], y=[top1_acc, top3_acc], palette=['#e74c3c', '#2ecc71'])
    
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Model Robustness: Top-1 vs Top-3\n{name}', fontsize=14)
    plt.ylim(0, 100)
    
    # Add text numbers on bars
    plt.text(0, top1_acc + 2, f"{top1_acc:.1f}%", ha='center', fontsize=14, weight='bold')
    plt.text(1, top3_acc + 2, f"{top3_acc:.1f}%", ha='center', fontsize=14, weight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(save_name, dpi=150)
    plt.close()

def plot_confidence_hist(correct_confs, wrong_confs, name, save_name):
    """Plots histograms of confidence scores for correct vs wrong predictions."""
    plt.figure(figsize=(10, 6))
    
    # Plot Correct in Green
    sns.histplot(correct_confs, color="green", label="Correct Predictions", kde=True, stat="density", alpha=0.5, bins=20)
    # Plot Wrong in Red
    sns.histplot(wrong_confs, color="red", label="Wrong Predictions", kde=True, stat="density", alpha=0.5, bins=20)

    plt.xlabel('Model Confidence (Probability Score)', fontsize=12)
    plt.ylabel('Density of Predictions', fontsize=12)
    plt.title(f'Prediction Confidence Distribution - {name}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_name, dpi=150)
    plt.close()

def plot_most_confused(cm, class_names, name, save_name, top_n=10):
    """Plots the specific pairs of food the model confuses most often."""
    # Zero out the diagonal (correct predictions) so we only see errors
    cm_errors = cm.copy()
    np.fill_diagonal(cm_errors, 0)
    
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm_errors[i, j]
            if count > 0:
                # (Actual Name, Predicted Name, Count)
                pairs.append((class_names[i], class_names[j], count))

    # Sort by count descending
    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = pairs[:top_n]

    if not top_pairs:
        print("   ⚠️ No errors found! Skipping 'Confused Pairs' plot.")
        return

    labels = [f"{p[0]} \nwas confused with\n{p[1]}" for p in top_pairs]
    counts = [p[2] for p in top_pairs]

    plt.figure(figsize=(10, 8))
    sns.barplot(x=counts, y=labels, palette="magma")
    
    plt.xlabel('Number of Misclassifications', fontsize=12)
    plt.title(f'Top {top_n} Most Confused Food Pairs - {name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    plt.close()

# --- MAIN EVALUATION FUNCTION ---

def evaluate_cuisine(name, cuisine_folder, split_folder_name, model_filename):
    print(f"\n" + "="*50)
    print(f"📊 GENERATING GRAPHS FOR: {name}")
    print("="*50)
    
    base_path = CUISINE_ROOT / cuisine_folder
    split_path = base_path / "data" / split_folder_name
    
    # Determine directory
    if (split_path / "test").exists():
        data_dir = split_path / "test"
    elif (split_path / "val").exists():
        data_dir = split_path / "val"
    else:
        print(f"❌ Error: Could not find 'test' or 'val' folder inside {split_path}")
        return

    # Load Classes
    classes_file = base_path / "models" / "classes.txt"
    if not classes_file.exists(): 
        print(f"❌ Classes file not found at {classes_file}")
        return
    class_names = [line.strip() for line in open(classes_file, "r")]
    num_classes = len(class_names)
    print(f"   📝 Classes found: {num_classes}")

    # Prepare Data
    dataset = datasets.ImageFolder(root=str(data_dir), transform=TFM)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load Model
    model_path = base_path / "models" / model_filename
    model = load_model(model_path, num_classes)
    if not model: return

    # --- INFERENCE LOOP ---
    print("   🚀 Running inference (calculating Top-1, Top-3, & Confidence)...")
    
    all_preds = []      # Top-1 predictions
    all_labels = []     # True labels
    
    # For Top-K
    top3_hits = 0
    total_samples = 0
    
    # For Confidence Histogram
    correct_confidences = []
    wrong_confidences = []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(imgs)
            
            # Convert to probabilities (Softmax)
            probs = F.softmax(logits, dim=1)
            
            # --- Top-1 Logic ---
            max_probs, preds = torch.max(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # --- Top-3 Logic ---
            # Get indices of top 3 probabilities
            _, top3_indices = probs.topk(3, dim=1, largest=True, sorted=True)
            
            # Check if true label is in top 3
            # labels.view(-1, 1) reshapes labels to [Batch, 1] to broadcast against [Batch, 3]
            current_hits = (top3_indices == labels.view(-1, 1)).sum().item()
            top3_hits += current_hits
            total_samples += labels.size(0)
            
            # --- Confidence Logic ---
            max_probs = max_probs.cpu().numpy()
            preds = preds.cpu().numpy()
            lbls = labels.cpu().numpy()
            
            for i in range(len(lbls)):
                if preds[i] == lbls[i]:
                    correct_confidences.append(max_probs[i])
                else:
                    wrong_confidences.append(max_probs[i])

    # --- METRICS CALCULATION ---
    acc = accuracy_score(all_labels, all_preds)
    top1_acc_percent = acc * 100
    top3_acc_percent = (top3_hits / total_samples) * 100
    
    print(f"   ✅ Top-1 Accuracy: {top1_acc_percent:.2f}%")
    print(f"   ✅ Top-3 Accuracy: {top3_acc_percent:.2f}%")

    # --- PLOTTING ---
    
    # Dynamic Sizing for Matrix
    plot_width = max(12, num_classes * 0.6)
    plot_height = max(10, num_classes * 0.6)
    font_scale = 0.6 if num_classes > 30 else 1.0

    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(plot_width, plot_height))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 8 * font_scale}, cbar=False)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(f'Confusion Matrix - {name}\nTop-1 Acc: {top1_acc_percent:.1f}%', fontsize=16)
    plt.xticks(rotation=90, fontsize=10 * font_scale)
    plt.yticks(rotation=0, fontsize=10 * font_scale)
    plt.tight_layout()
    plt.savefig(f"{name.replace(' ', '_')}_1_Confusion_Matrix.png", dpi=150)
    print(f"   📸 Saved: Confusion Matrix")
    plt.close()

    # 2. Per-Class Accuracy Bar Plot
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    per_class_acc = cm_normalized.diagonal()
    plt.figure(figsize=(plot_width, 8))
    sns.barplot(x=class_names, y=per_class_acc, palette="viridis")
    plt.xlabel('Dish Name', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Per-Class Accuracy - {name}', fontsize=14)
    plt.xticks(rotation=90, fontsize=9 * font_scale)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(f"{name.replace(' ', '_')}_2_PerClass_Accuracy.png", dpi=150)
    print(f"   📸 Saved: Per-Class Accuracy")
    plt.close()

    # 3. Top-K Accuracy Comparison
    plot_top_k_bar(top1_acc_percent, top3_acc_percent, name, 
                   f"{name.replace(' ', '_')}_3_TopK_Accuracy.png")
    print(f"   📸 Saved: Top-K Accuracy Graph")

    # 4. Confidence Histogram
    plot_confidence_hist(correct_confidences, wrong_confidences, name,
                         f"{name.replace(' ', '_')}_4_Confidence_Hist.png")
    print(f"   📸 Saved: Confidence Histogram")

    # 5. Top Confused Pairs
    plot_most_confused(cm, class_names, name,
                       f"{name.replace(' ', '_')}_5_Confused_Pairs.png")
    print(f"   📸 Saved: Confused Pairs Graph")


if __name__ == "__main__":
    # Ensure you are pointing to the correct filenames
    # Example 1: Punjabi
    evaluate_cuisine("Punjabi", "Punjabi Cuisines", "Punjabi_Split_Data", "best_efficientnet_b0.pth")
    
    # Example 2: South Indian
    evaluate_cuisine("South Indian", "South Indian Cuisines", "SouthIndian_Split_Data", "best_b3.pth")
    
    print("\n✨ All Graphs Generated Successfully!")





# import os
# import torch
# import torch.nn as nn
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from torchvision import transforms, datasets
# from torch.utils.data import DataLoader
# from sklearn.metrics import confusion_matrix, accuracy_score


# ROOT = Path(__file__).resolve().parent
# CUISINE_ROOT = ROOT / "cuisines"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(f"⚙️  Running on device: {device}")

# TFM = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225]),
# ])


# def load_model(path, num_classes):
#     path = str(path)
#     if "b3" in path.lower():
#         print(f"   Model detected: EfficientNet B3")
#         from torchvision.models import efficientnet_b3
#         m = efficientnet_b3(weights=None) 
#         m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
#     else:
#         print(f"   Model detected: EfficientNet B0")
#         from torchvision.models import efficientnet_b0
#         m = efficientnet_b0(weights=None)
#         m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

#     try:
#         state = torch.load(path, map_location=device)
#         m.load_state_dict(state)
#         m.to(device).eval()
#         return m
#     except Exception as e:
#         print(f"❌ Error loading model at {path}: {e}")
#         return None


# def evaluate_cuisine(name, cuisine_folder, split_folder_name, model_filename):
#     print(f"\n" + "="*50)
#     print(f"📊 GENERATING GRAPHS FOR: {name}")
#     print("="*50)
    
   
#     base_path = CUISINE_ROOT / cuisine_folder
#     split_path = base_path / "data" / split_folder_name
    
#     if (split_path / "test").exists():
#         data_dir = split_path / "test"
#     elif (split_path / "val").exists():
#         data_dir = split_path / "val"
#     else:
#         print(f"❌ Error: Could not find 'test' or 'val' folder inside {split_path}")
#         return


#     classes_file = base_path / "models" / "classes.txt"
#     if not classes_file.exists(): return
#     class_names = [line.strip() for line in open(classes_file, "r")]
#     num_classes = len(class_names)
#     print(f"   📝 Classes found: {num_classes}")

  
#     dataset = datasets.ImageFolder(root=str(data_dir), transform=TFM)
#     loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    
#     model_path = base_path / "models" / model_filename
#     model = load_model(model_path, num_classes)
#     if not model: return


#     print("   🚀 Running inference...")
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for imgs, labels in loader:
#             imgs = imgs.to(device)
#             outputs = model(imgs)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     acc = accuracy_score(all_labels, all_preds)
#     print(f"   ✅ Accuracy: {acc*100:.2f}%")

#     # -------------------------------------------------------
#     # DYNAMIC SIZING LOGIC
#     # Calculates plot size based on number of dishes
#     # -------------------------------------------------------
#     # Scale factor: 0.6 inches per class ensures readable labels
#     # Minimum size: 12x10 inches
#     plot_width = max(12, num_classes * 0.6)
#     plot_height = max(10, num_classes * 0.6)
    
#     font_scale = 1.0
#     if num_classes > 30: font_scale = 0.6  # Shrink text for many classes

#     # --- F. Plot Confusion Matrix ---
#     cm = confusion_matrix(all_labels, all_preds)
    
#     plt.figure(figsize=(plot_width, plot_height))
    
#     # Create Heatmap
#     # annot_kws controls the size of the numbers inside the boxes
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=dataset.classes, 
#                 yticklabels=dataset.classes,
#                 annot_kws={"size": 8 * font_scale}, 
#                 cbar=False) # Turn off colorbar to save space if needed
    
#     plt.xlabel('Predicted Label', fontsize=14)
#     plt.ylabel('True Label', fontsize=14)
#     plt.title(f'Confusion Matrix - {name}\nAccuracy: {acc*100:.1f}%', fontsize=16)
    
#     # Rotate x-labels 90 degrees perfectly vertical
#     plt.xticks(rotation=90, fontsize=10 * font_scale)
#     plt.yticks(rotation=0, fontsize=10 * font_scale)
    
#     plt.tight_layout()
#     save_name_cm = f"{name.replace(' ', '_')}_Confusion_Matrix.png"
#     plt.savefig(save_name_cm, dpi=150) # Higher DPI for clearer text
#     print(f"   📸 Saved High-Res Matrix: {save_name_cm}")
#     plt.close()

#     # --- G. Plot Per-Class Accuracy ---
#     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     per_class_acc = cm_normalized.diagonal()
    
#     plt.figure(figsize=(plot_width, 8)) # Wide but short for bar chart
#     sns.barplot(x=dataset.classes, y=per_class_acc, palette="viridis")
    
#     plt.xlabel('Dish Name', fontsize=12)
#     plt.ylabel('Accuracy', fontsize=12)
#     plt.title(f'Per-Class Accuracy - {name}', fontsize=14)
#     plt.xticks(rotation=90, fontsize=9 * font_scale)
#     plt.ylim(0, 1.0)
#     plt.tight_layout()
    
#     save_name_acc = f"{name.replace(' ', '_')}_Accuracy.png"
#     plt.savefig(save_name_acc, dpi=150)
#     print(f"   📸 Saved High-Res Accuracy: {save_name_acc}")
#     plt.close()

# if __name__ == "__main__":
#     evaluate_cuisine("Punjabi", "Punjabi Cuisines", "Punjabi_Split_Data", "best_efficientnet_b0.pth")
#     evaluate_cuisine("South Indian", "South Indian Cuisines", "SouthIndian_Split_Data", "best_b3.pth")
#     print("\n✨ Graphs generated. Open the PNG files to see the clean versions.")