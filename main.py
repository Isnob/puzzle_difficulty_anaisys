import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from tqdm import tqdm

# =========================
# 1. Загрузка таблицы
# =========================
df = pd.read_csv("data.csv")

# =========================
# 2. Где лежат картинки
# =========================
# Вариант A: картинки прямо рядом со скриптом / ноутбуком
# IMAGE_DIR = Path(".")

# Вариант B: если картинки лежат в подпапке, раскомментируй:
IMAGE_DIR = Path("daily-rebuses")

# =========================
# 3. Предобученная ResNet
# =========================
weights = ResNet18_Weights.DEFAULT
base_model = resnet18(weights=weights)
model = torch.nn.Sequential(*list(base_model.children())[:-1])  # 512 признаков
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])

# =========================
# 4. Функции
# =========================
def resolve_local_image_path(img_url: str, image_dir: Path) -> Path | None:
    """
    Из /daily-rebuses/1_v1.jpg делаем локальный путь:
    - ./1_v1.jpg
    или
    - ./daily-rebuses/1_v1.jpg
    """
    if pd.isna(img_url):
        return None

    filename = os.path.basename(str(img_url))  # 1_v1.jpg
    candidate1 = image_dir / filename
    candidate2 = image_dir / str(img_url).lstrip("/")  # daily-rebuses/1_v1.jpg

    if candidate1.exists():
        return candidate1
    if candidate2.exists():
        return candidate2

    return None

def extract_image_features(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        feats = model(x).squeeze()

    return feats.cpu().numpy()

# =========================
# 5. Извлечение признаков
# =========================
image_features = []
resolved_paths = []
missing_paths = []

for img_url in tqdm(df["img_url"], desc="Извлечение признаков"):
    local_path = resolve_local_image_path(img_url, IMAGE_DIR)
    resolved_paths.append(str(local_path) if local_path is not None else None)

    if local_path is None:
        image_features.append(np.zeros(512, dtype=np.float32))
        missing_paths.append(img_url)
    else:
        image_features.append(extract_image_features(local_path).astype(np.float32))

image_features = np.vstack(image_features)

# =========================
# 6. Добавление в таблицу
# =========================
img_cols = [f"img_feat_{i}" for i in range(512)]
img_df = pd.DataFrame(image_features, columns=img_cols)

df["resolved_img_path"] = resolved_paths
df = pd.concat([df, img_df], axis=1)

# =========================
# 7. Проверка, что всё реально подцепилось
# =========================
zero_rows = (df[img_cols].abs().sum(axis=1) == 0).sum()
print(f"Строк с нулевыми image features: {zero_rows} из {len(df)}")

if missing_paths:
    print("\nНе найдено файлов:", len(missing_paths))
    print("Первые 10 пропусков:")
    for p in missing_paths[:10]:
        print(p)

# =========================
# 8. Сохранение
# =========================
df.to_csv("data_with_image_features_fixed.csv", index=False)
print("\nСохранено: data_with_image_features_fixed.csv")