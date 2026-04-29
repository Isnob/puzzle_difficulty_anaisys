import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# =========================================================
# 1. Конфигурация
# =========================================================
INPUT_CSV = "data.csv"
OUTPUT_CSV = "data_with_efficientnet_features.csv"

# Где лежат картинки:
# Вариант 1: картинки рядом со скриптом / ноутбуком
IMAGE_DIR = Path(".")

# Если картинки в отдельной папке, например daily-rebuses, то можно так:
# IMAGE_DIR = Path("daily-rebuses")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# 2. Загрузка таблицы
# =========================================================
df = pd.read_csv(INPUT_CSV)

# =========================================================
# 3. Модель EfficientNet-B0 без классификационной головы
# =========================================================
weights = EfficientNet_B0_Weights.DEFAULT
base_model = efficientnet_b0(weights=weights)

# Оставляем только extractor признаков:
# features -> avgpool -> flatten
model = torch.nn.Sequential(
    base_model.features,
    base_model.avgpool,
    torch.nn.Flatten()
).to(DEVICE)

model.eval()

transform = weights.transforms()

# =========================================================
# 4. Функции
# =========================================================
def resolve_local_image_path(img_url: str, image_dir: Path) -> Path | None:
    """
    Пример:
    /daily-rebuses/1_v1.jpg -> ./1_v1.jpg
    или -> ./daily-rebuses/1_v1.jpg
    """
    if pd.isna(img_url):
        return None

    img_url = str(img_url)
    filename = os.path.basename(img_url)

    candidate1 = image_dir / filename
    candidate2 = image_dir / img_url.lstrip("/")

    if candidate1.exists():
        return candidate1
    if candidate2.exists():
        return candidate2

    return None


def extract_image_features(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feats = model(x)

    return feats.squeeze().cpu().numpy().astype(np.float32)


# =========================================================
# 5. Извлечение признаков
# =========================================================
all_features = []
resolved_paths = []
missing_paths = []

for img_url in tqdm(df["img_url"], desc="Извлечение EfficientNet-признаков"):
    local_path = resolve_local_image_path(img_url, IMAGE_DIR)
    resolved_paths.append(str(local_path) if local_path is not None else None)

    if local_path is None:
        # EfficientNet-B0 после avgpool даёт 1280 признаков
        all_features.append(np.zeros(1280, dtype=np.float32))
        missing_paths.append(img_url)
    else:
        try:
            feats = extract_image_features(local_path)
            all_features.append(feats)
        except Exception as e:
            print(f"Ошибка при обработке {local_path}: {e}")
            all_features.append(np.zeros(1280, dtype=np.float32))
            missing_paths.append(img_url)

all_features = np.vstack(all_features)

# =========================================================
# 6. Добавляем признаки в DataFrame
# =========================================================
feature_cols = [f"effnet_feat_{i}" for i in range(all_features.shape[1])]
features_df = pd.DataFrame(all_features, columns=feature_cols)

df["resolved_img_path"] = resolved_paths
df = pd.concat([df, features_df], axis=1)

# =========================================================
# 7. Проверка
# =========================================================
zero_rows = (df[feature_cols].abs().sum(axis=1) == 0).sum()
print(f"\nСтрок с полностью нулевыми EfficientNet-признаками: {zero_rows} из {len(df)}")

if missing_paths:
    print(f"\nНе найдено / не обработано файлов: {len(missing_paths)}")
    print("Первые 10 проблемных путей:")
    for p in missing_paths[:10]:
        print(p)

# =========================================================
# 8. Сохранение
# =========================================================
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nГотово. Файл сохранён: {OUTPUT_CSV}")
