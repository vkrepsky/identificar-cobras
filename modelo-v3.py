import argparse
import json
import os
import shutil
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Configurações globais
BATCH_SIZE = 32  # Tamanho do lote para treinamento
NUM_EPOCHS = 30  # Aumentado de 30 para 60 épocas
LEARNING_RATE = 0.0001  # Taxa de aprendizado
IMAGE_SIZE = 224  # Tamanho padrão para imagens (224x224 é padrão para redes pré-treinadas)
NUM_WORKERS = 4  # Número de workers para carregamento de dados
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_IMAGES_PER_CLASS = 500  # Aumentado de 300 para 500 imagens por classe após augmentation
WEIGHT_DECAY = 1e-4  # Adicionado weight decay para regularização

# Diretórios para salvar resultados
MODELS_DIR = "models"  # Para modelos treinados
PLOTS_DIR = "plots"  # Para gráficos
REPORTS_DIR = "reports"  # Para relatórios e métricas
ENSEMBLE_DIR = "ensemble"  # Para modelos de ensemble


def check_augmented_data(augmented_dir):
    """
    Verifica se existe um conjunto de dados aumentados válido no diretório especificado.

    Args:
        augmented_dir: Caminho para o diretório de dados aumentados

    Returns:
        bool: True se os dados aumentados existem e são válidos, False caso contrário
    """
    # Verificar se o diretório existe
    if not os.path.exists(augmented_dir) or not os.path.isdir(augmented_dir):
        print(f"❌ Diretório de dados aumentados não encontrado: {augmented_dir}")
        return False

    # Verificar se há subdiretórios (classes)
    classes = [d for d in os.listdir(augmented_dir) if os.path.isdir(os.path.join(augmented_dir, d))]
    if not classes:
        print(f"❌ Diretório de dados aumentados não contém classes: {augmented_dir}")
        return False

    print(f"📁 Encontradas {len(classes)} classes em {augmented_dir}")

    # Verificar se há imagens em pelo menos algumas classes
    total_images = 0
    classes_with_images = 0

    for class_name in classes:
        class_dir = os.path.join(augmented_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        num_images = len(images)

        if num_images > 0:
            classes_with_images += 1
            total_images += num_images
            print(f"  - Classe '{class_name}': {num_images} imagens")

    if classes_with_images == 0:
        print(f"❌ Nenhuma imagem encontrada em nenhuma classe em {augmented_dir}")
        return False

    print(f"✅ Encontradas {total_images} imagens em {classes_with_images}/{len(classes)} classes")
    return True

def create_output_directories(output_base_dir):
    """
    Cria os diretórios necessários para salvar os resultados.
    """
    # Convertendo para Path para melhor manipulação
    base_path = Path(output_base_dir)

    # Define os diretórios
    directories = {
        'base': str(base_path),
        'models': str(base_path / MODELS_DIR),
        'plots': str(base_path / PLOTS_DIR),
        'reports': str(base_path / REPORTS_DIR),
        'ensemble': str(base_path / ENSEMBLE_DIR)  # Adiciona diretório para os modelos de ensemble
    }

    # Cria cada diretório se não existir
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Diretório criado/verificado: {dir_path}")

    return directories


class SnakeDataset(Dataset):
    """
    Dataset para as imagens de serpentes.
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


class SnakeCNN(nn.Module):
    """
    Modelo CNN original do artigo com regularização aumentada
    """

    def __init__(self, num_classes):
        super(SnakeCNN, self).__init__()

        # Extrator de características com 3 blocos convolucionais
        self.features = nn.Sequential(
            # Primeiro bloco
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # Adicionado BatchNorm
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Segundo bloco
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Adicionado BatchNorm
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Terceiro bloco
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Adicionado BatchNorm
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Adicionado quarto bloco para maior capacidade
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Para uma imagem 224x224, após 4 max poolings, temos 14x14
        fc_input_size = 128 * 14 * 14

        # Classificador com dropout aumentado
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 256),  # Aumentado de 128 para 256
            nn.BatchNorm1d(256),  # Adicionado BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Aumentado de 0.2 para 0.5
            nn.Linear(256, 128),  # Adicionado camada intermediária
            nn.BatchNorm1d(128),  # Adicionado BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Adicionado dropout nesta camada também
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SnakeCNN_TransferLearning(nn.Module):
    """
    Modelo para classificação de serpentes usando transfer learning
    com diferentes backbones pré-treinados, mantendo a estrutura do
    classificador igual ao SnakeCNN original, mas com regularização aprimorada.
    """

    def __init__(self, num_classes, base_model='resnet50', freeze_backbone=True):
        super(SnakeCNN_TransferLearning, self).__init__()

        self.base_model_name = base_model
        self.freeze_backbone = freeze_backbone

        # Seleção do modelo base pré-treinado
        if base_model == 'resnet50':
            base = models.resnet50(weights='IMAGENET1K_V1')
            backbone = nn.Sequential(*list(base.children())[:-2])  # Remove avg pool e FC
            feature_size = 2048  # ResNet50 produz 2048 canais de features

        elif base_model == 'efficientnet_b3':
            base = models.efficientnet_b3(weights='IMAGENET1K_V1')
            backbone = base.features  # Extrator de características
            feature_size = 1536  # EfficientNet-B3 produz 1536 canais

        elif base_model == 'densenet169':
            base = models.densenet169(weights='IMAGENET1K_V1')
            backbone = base.features
            feature_size = 1664  # DenseNet169 produz 1664 canais

        elif base_model == 'inception_v3':
            base = models.inception_v3(weights='IMAGENET1K_V1')
            backbone = nn.Sequential(
                base.Conv2d_1a_3x3, base.Conv2d_2a_3x3, base.Conv2d_2b_3x3,
                base.maxpool1, base.Conv2d_3b_1x1, base.Conv2d_4a_3x3,
                base.maxpool2, base.Mixed_5b, base.Mixed_5c, base.Mixed_5d,
                base.Mixed_6a, base.Mixed_6b, base.Mixed_6c, base.Mixed_6d,
                base.Mixed_6e, base.Mixed_7a, base.Mixed_7b, base.Mixed_7c
            )
            feature_size = 2048  # Inception_v3 produz 2048 canais

        elif base_model == 'vgg16':
            base = models.vgg16(weights='IMAGENET1K_V1')
            backbone = base.features  # Extrator de características VGG
            feature_size = 512  # VGG16 produz 512 canais

        else:
            raise ValueError(f"Modelo base não suportado: {base_model}")

        # Congelar os pesos do backbone pré-treinado
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        self.backbone = backbone

        # Camada de pooling adaptativa para garantir dimensões consistentes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classificador aprimorado com maior regularização
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 512),  # Aumentado para 512 neurônios
            nn.BatchNorm1d(512),  # Adicionado BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Aumentado de 0.2 para 0.5
            nn.Linear(512, 256),  # Adicionado camada intermediária
            nn.BatchNorm1d(256),  # Adicionado BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extração de features pelo backbone pré-treinado
        x = self.backbone(x)
        # Adaptação do tamanho
        x = self.adaptive_pool(x)
        # Classificação seguindo a estrutura do SnakeCNN original
        x = self.classifier(x)
        return x


# Implementação do U-Net (adaptado para classificação)
class UNetEncoder(nn.Module):
    """Encoder do U-Net adaptado para classificação com BatchNorm adicionado"""

    def __init__(self):
        super(UNetEncoder, self).__init__()
        # Camadas do encoder do U-Net
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        bottleneck = self.bottleneck(p4)
        return bottleneck


class SnakeCNN_UNet(nn.Module):
    """
    Modelo de classificação de serpentes usando o encoder do U-Net
    com o classificador do SnakeCNN original
    """

    def __init__(self, num_classes):
        super(SnakeCNN_UNet, self).__init__()

        # Usar o encoder do U-Net
        self.backbone = UNetEncoder()

        # Congelar metade do backbone para simular transfer learning
        backbone_params = list(self.backbone.parameters())
        half_point = len(backbone_params) // 2
        for param in backbone_params[:half_point]:
            param.requires_grad = False

        # Adaptação para extrair features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classificador aprimorado com mais regularização
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),  # U-Net bottleneck tem 1024 canais
            nn.BatchNorm1d(512),  # Adicionado BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Aumentado dropout
            nn.Linear(512, 256),  # Adicionado camada intermediária
            nn.BatchNorm1d(256),  # Adicionado BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


def get_snake_model_with_transfer_learning(num_classes, base_model='resnet50', freeze_backbone=True):
    """
    Retorna um modelo SnakeCNN usando transfer learning.
    """
    if base_model == 'unet':
        return SnakeCNN_UNet(num_classes)
    else:
        return SnakeCNN_TransferLearning(num_classes, base_model, freeze_backbone)


def augment_and_save_images(data_dir, output_dir, target_images_per_class):
    """
    Gera imagens aumentadas para cada classe até atingir o número alvo por classe.
    Implementa uma augmentação mais agressiva para melhorar a generalização.
    """
    print("\n" + "=" * 80)
    print("🔄 AUMENTAÇÃO DE DADOS - GERANDO IMAGENS SINTÉTICAS 🔄".center(80))
    print("=" * 80)

    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Transformações avançadas para data augmentation
    augmentation_transforms = [
        transforms.RandomHorizontalFlip(p=0.7),  # Aumentar probabilidade
        transforms.RandomVerticalFlip(p=0.7),  # Importante para serpentes (podem aparecer em qualquer orientação)
        transforms.RandomRotation(60),  # Ângulos maiores
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Variações de cor mais intensas
        transforms.RandomAffine(degrees=30, translate=(0.3, 0.3), scale=(0.7, 1.3)),  # Mais deformações
        transforms.RandomPerspective(distortion_scale=0.6, p=0.7),
        transforms.GaussianBlur(kernel_size=5),  # Blur mais forte
        transforms.RandomGrayscale(p=0.1),  # Adicionar variação monocromática ocasional
    ]

    print(f"\n📊 Alvo: {target_images_per_class} imagens por classe após aumentação")
    print(f"📁 Diretório original: {data_dir}")
    print(f"📁 Diretório de saída: {output_dir}")

    # Processar cada classe
    class_stats = []
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Criar diretório da classe no output
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        # Listar imagens originais
        original_images = [img for img in os.listdir(class_dir)
                           if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"\n📸 Classe: {class_name}")
        print(f"  ↳ Imagens originais: {len(original_images)}")

        # Se não há imagens originais, pular
        if len(original_images) == 0:
            print(f"  ⚠️ Nenhuma imagem encontrada para a classe {class_name}. Pulando.")
            continue

        # Copiar imagens originais para o novo diretório
        for img_name in original_images:
            src_path = os.path.join(class_dir, img_name)
            dst_path = os.path.join(output_class_dir, f"original_{img_name}")
            shutil.copy2(src_path, dst_path)

        # Se já temos imagens suficientes, pular aumentação
        if len(original_images) >= target_images_per_class:
            print(f"  ✅ Já possui {len(original_images)} imagens (>= {target_images_per_class}). Pulando aumentação.")
            class_stats.append((class_name, len(original_images), len(original_images), 0))
            continue

        # Calcular quantas imagens aumentadas precisamos gerar por imagem original
        num_augmentations_per_image = (target_images_per_class - len(original_images)) // len(original_images)
        remaining_augmentations = (target_images_per_class - len(original_images)) % len(original_images)

        print(f"  ↳ Gerando ~{num_augmentations_per_image} variações por imagem original")

        # Contador para imagens geradas
        generated_count = 0

        # Gerar imagens aumentadas
        for i, img_name in enumerate(original_images):
            img_path = os.path.join(class_dir, img_name)

            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"  ⚠️ Erro ao abrir a imagem {img_name}: {e}")
                continue

            # Número de augmentations para esta imagem
            num_aug = num_augmentations_per_image
            if i < remaining_augmentations:
                num_aug += 1

            # Gerar versões aumentadas
            for aug_idx in range(num_aug):
                aug_img = img.copy()

                # Aplicar transformações aleatórias - mais transformações por imagem
                random.shuffle(augmentation_transforms)
                num_transforms = random.randint(3,
                                                min(5, len(augmentation_transforms)))  # Mais transformações por imagem

                for t in range(num_transforms):
                    try:
                        aug_img = augmentation_transforms[t](aug_img)
                    except Exception as e:
                        print(f"  ⚠️ Erro ao aplicar transformação em {img_name}: {e}")
                        continue

                # Salvar imagem aumentada
                try:
                    output_path = os.path.join(output_class_dir, f"aug_{aug_idx}_{img_name}")
                    aug_img.save(output_path)
                    generated_count += 1
                except Exception as e:
                    print(f"  ⚠️ Erro ao salvar a imagem aumentada {img_name}: {e}")
                    continue

        # Verificar número final de imagens
        final_count = len(os.listdir(output_class_dir))
        print(
            f"  ✅ Aumentação concluída: {final_count} imagens ({len(original_images)} originais + {generated_count} geradas)")
        class_stats.append((class_name, len(original_images), final_count, generated_count))

    # Resumo final
    print("\n" + "=" * 80)
    print("📊 RESUMO DA AUMENTAÇÃO DE DADOS 📊".center(80))
    print("=" * 80)
    print(f"{'Classe':<20} {'Originais':<10} {'Geradas':<10} {'Total':<10} {'% Aumento':<10}")
    print("-" * 80)

    for class_name, original_count, final_count, generated_count in class_stats:
        percent_increase = (final_count / original_count - 1) * 100 if original_count > 0 else 0
        print(f"{class_name:<20} {original_count:<10} {generated_count:<10} {final_count:<10} {percent_increase:.1f}%")

    print("-" * 80)
    total_original = sum(stats[1] for stats in class_stats)
    total_final = sum(stats[2] for stats in class_stats)
    total_generated = sum(stats[3] for stats in class_stats)
    total_percent = (total_final / total_original - 1) * 100 if total_original > 0 else 0
    print(f"{'TOTAL':<20} {total_original:<10} {total_generated:<10} {total_final:<10} {total_percent:.1f}%")

    print("\n✅ Aumentação de dados concluída com sucesso!")

    return output_dir


def prepare_data_with_test_set(data_dir, min_images_per_class=20):
    """
    Prepara os dados para treinamento, validação e teste, organizando caminhos e rótulos.
    Divide os dados em três conjuntos: treinamento, validação e teste.
    """
    image_paths = []
    labels = []
    class_names = []
    class_counts = {}
    valid_class_dirs = []

    # Primeiro, conta as imagens em cada classe para determinar quais classes são válidas
    print("\nContando imagens em cada classe:")
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            image_count = sum(1 for img_name in os.listdir(class_dir)
                              if img_name.lower().endswith(('.png', '.jpg', '.jpeg')))

            class_counts[class_name] = image_count
            print(f"Classe {class_name}: {image_count} imagens")

            # Verifica se a classe tem o número mínimo de imagens
            if image_count >= min_images_per_class:
                valid_class_dirs.append((class_name, class_dir))
            else:
                print(f"⚠️ Ignorando classe {class_name} por ter menos de {min_images_per_class} imagens")

    # Depois, processa apenas as classes válidas
    class_to_idx = {}  # Mapeamento de nome da classe para índice

    print("\nClasses incluídas no treinamento:")
    for idx, (class_name, class_dir) in enumerate(valid_class_dirs):
        class_names.append(class_name)
        class_to_idx[class_name] = idx

        # Adiciona as imagens e seus rótulos
        class_image_paths = [os.path.join(class_dir, img_name)
                             for img_name in os.listdir(class_dir)
                             if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))]

        image_paths.extend(class_image_paths)
        labels.extend([idx] * len(class_image_paths))

        print(f"Classe {class_name} (índice {idx}): {len(class_image_paths)} imagens")

    print(f"\nTotal de classes válidas: {len(class_names)}")
    print(f"Total de imagens: {len(image_paths)}")

    # Se não houver classes válidas, levanta um erro
    if len(class_names) == 0:
        raise ValueError(f"Nenhuma classe encontrada com pelo menos {min_images_per_class} imagens!")

    # Dividir em treino, validação e teste (60%, 20%, 20%)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.4, random_state=42, stratify=labels
    )

    # Dividir os dados restantes em validação e teste
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    print(f"\nDivisão dos dados:")
    print(f"  - Treinamento: {len(train_paths)} imagens")
    print(f"  - Validação: {len(val_paths)} imagens")
    print(f"  - Teste: {len(test_paths)} imagens")

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names, class_to_idx


def get_advanced_transforms():
    """
    Retorna transformações avançadas para modelos pré-treinados com mais data augmentation.
    """
    # Transformações de treino mais diversas para reduzir overfitting
    train_transform = transforms.Compose([
        # Redimensiona para o tamanho esperado pelos modelos pré-treinados
        transforms.Resize((224, 224)),
        # Data augmentation aprimorada
        transforms.RandomHorizontalFlip(p=0.7),
        transforms.RandomVerticalFlip(p=0.5),  # Útil para serpentes que podem aparecer em qualquer orientação
        transforms.RandomRotation(60),  # Ângulos maiores
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Mais variações de cor
        transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.4),
        transforms.RandomGrayscale(p=0.05),
        # Conversão para tensor
        transforms.ToTensor(),
        # Normalização específica para modelos pré-treinados no ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform, test_transform


def generate_learning_curves(model_name, history, output_dir):
    """
    Gera e salva as curvas de aprendizado para um modelo.
    """
    plt.figure(figsize=(14, 5))

    # Plot de acurácia
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], 'b-', label='Treino')
    plt.plot(history['val_acc'], 'r-', label='Validação')
    plt.title(f'Acurácia de Treinamento - {model_name}')
    plt.xlabel('Época')
    plt.ylabel('Acurácia (%)')
    plt.legend()
    plt.grid(True)

    # Plot de perda
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], 'b-', label='Treino')
    plt.plot(history['val_loss'], 'r-', label='Validação')
    plt.title(f'Perda de Treinamento - {model_name}')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close()


def generate_comparative_charts(comparative_results, output_dir):
    """
    Gera gráficos comparativos entre os modelos.
    """
    metrics = ['accuracy', 'training_time', 'inference_time', 'model_size_mb']
    titles = ['Acurácia (%)', 'Tempo de Treinamento (min)', 'Tempo de Inferência (ms)', 'Tamanho do Modelo (MB)']

    plt.figure(figsize=(20, 15))

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(2, 2, i + 1)
        data = comparative_results[metric]

        # Ordenar por desempenho
        if metric == 'accuracy':
            # Maior é melhor
            sorted_data = {k: v for k, v in sorted(data.items(), key=lambda item: item[1], reverse=True)}
        else:
            # Menor é melhor
            sorted_data = {k: v for k, v in sorted(data.items(), key=lambda item: item[1])}

        # Plotar
        bars = plt.bar(range(len(sorted_data)), list(sorted_data.values()), color='skyblue')
        plt.xticks(range(len(sorted_data)), list(sorted_data.keys()), rotation=45)
        plt.title(title)
        plt.tight_layout()

        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom')

    plt.savefig(os.path.join(output_dir, 'comparative_metrics.png'))
    plt.close()

    # Tabela comparativa
    comparison_df = pd.DataFrame({
        'Acurácia (%): Validação': comparative_results['accuracy'],
        'Acurácia (%): Teste': comparative_results.get('test_accuracy', {}),
        'Tempo de Treinamento (min)': comparative_results['training_time'],
        'Tempo de Inferência (ms)': comparative_results['inference_time'],
        'Tamanho do Modelo (MB)': comparative_results['model_size_mb']
    })

    # Salvar como CSV
    comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'))

    # Plotar como tabela gráfica
    plt.figure(figsize=(14, len(comparison_df) * 0.8))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    table = plt.table(cellText=comparison_df.values.round(2),
                      rowLabels=comparison_df.index,
                      colLabels=comparison_df.columns,
                      cellLoc='center',
                      loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.savefig(os.path.join(output_dir, 'comparison_table.png'), bbox_inches='tight')
    plt.close()


class NumpyEncoder(json.JSONEncoder):
    """Classe para serializar arrays numpy em JSON"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def train_with_progressive_unfreezing(model, criterion, optimizer, train_loader, val_loader, device, epochs,
                                      class_weights=None, model_name=""):
    """
    Treina o modelo com descongelamento progressivo do backbone para melhorar a generalização.

    Args:
        model: Modelo a ser treinado
        criterion: Função de perda
        optimizer: Otimizador
        train_loader: DataLoader para o conjunto de treinamento
        val_loader: DataLoader para o conjunto de validação
        device: Dispositivo onde os cálculos serão realizados (CPU/GPU)
        epochs: Número total de épocas
        class_weights: Pesos de classe para lidar com desbalanceamentos
        model_name: Nome do modelo para logs

    Returns:
        history: Histórico de treinamento
        best_model_state: Estado do melhor modelo (weights)
        best_val_acc: Melhor acurácia de validação
    """
    # Inicializar histórico
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    # Melhor modelo
    best_val_acc = 0.0
    best_model_state = None

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    # Verificar se esse modelo usa transfer learning
    has_backbone = hasattr(model, 'backbone') and hasattr(model, 'freeze_backbone')

    # Definir épocas para descongelamento progressivo
    first_unfreeze = epochs // 3  # Descongelar as últimas camadas após 1/3 das épocas
    second_unfreeze = 2 * epochs // 3  # Descongelar completamente após 2/3 das épocas

    print(f"\n{'=' * 80}")
    print(f"🔥 Treinando modelo: {model_name} usando descongelamento progressivo")
    print(f"{'=' * 80}")

    for epoch in range(epochs):
        # Modificar quais partes do modelo são treináveis com base na época atual
        if has_backbone:
            if epoch == first_unfreeze:
                # Fase 2: Descongelar as últimas camadas do backbone
                print(f"\n📢 Época {epoch + 1}: Descongelando as últimas camadas do backbone...")

                # Para redes como ResNet, VGG, etc - descongelamos diferentes partes dependendo da arquitetura
                if hasattr(model.backbone, "children"):
                    # Obter todas as camadas
                    backbone_layers = list(model.backbone.children())

                    # Descongelar os últimos 30% do backbone
                    num_layers = len(backbone_layers)
                    for i, layer in enumerate(backbone_layers):
                        if i >= int(0.7 * num_layers):  # Descongelar últimos 30%
                            for param in layer.parameters():
                                param.requires_grad = True

                # Ajustar taxa de aprendizado
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1  # Reduzir LR para fine-tuning

                print(f"📊 Nova taxa de aprendizado: {optimizer.param_groups[0]['lr']}")

            elif epoch == second_unfreeze:
                # Fase 3: Descongelar todo o backbone
                print(f"\n📢 Época {epoch + 1}: Descongelando todo o backbone...")

                # Descongelar todos os parâmetros
                for param in model.backbone.parameters():
                    param.requires_grad = True

                # Ajustar taxa de aprendizado ainda mais
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1  # Reduzir mais a LR

                print(f"📊 Nova taxa de aprendizado: {optimizer.param_groups[0]['lr']}")

        # Fase de treinamento
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Usar label smoothing via função de perda - já implementado no critério passado
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Fase de validação
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        # Atualizar scheduler
        scheduler.step(val_loss)

        # Atualizar histórico
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Salvar o melhor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"✅ Novo melhor modelo salvo! Acurácia de validação: {val_acc:.2f}%")

        # Imprimir progresso
        print(f'Época {epoch + 1}/{epochs} | '
              f'Treino: Perda {train_loss:.4f}, Acc {train_acc:.2f}% | '
              f'Val: Perda {val_loss:.4f}, Acc {val_acc:.2f}%')

    return history, best_model_state, best_val_acc


def evaluate_on_test_set(model, test_loader, criterion, device):
    """
    Avalia o modelo no conjunto de teste.

    Args:
        model: Modelo a ser avaliado
        test_loader: DataLoader para o conjunto de teste
        criterion: Função de perda
        device: Dispositivo onde os cálculos serão realizados (CPU/GPU)

    Returns:
        test_acc: Acurácia no conjunto de teste
        test_loss: Perda no conjunto de teste
        y_true: Rótulos verdadeiros
        y_pred: Previsões
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * test_correct / test_total

    return test_acc, test_loss, y_true, y_pred


def ensemble_prediction(models, input_tensor, device):
    """
    Faz previsão usando um conjunto (ensemble) de modelos.

    Args:
        models: Lista de modelos a serem usados para previsão
        input_tensor: Tensor de entrada
        device: Dispositivo onde os cálculos serão realizados (CPU/GPU)

    Returns:
        probabilities: Probabilidades médias previstas pelo ensemble
    """
    all_outputs = []

    with torch.no_grad():
        for model in models:
            model.eval()
            outputs = model(input_tensor.to(device))
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            all_outputs.append(probabilities)

    # Média das probabilidades de todos os modelos
    ensemble_output = torch.stack(all_outputs).mean(dim=0)

    return ensemble_output


def create_ensemble_model(trained_models, model_paths, class_names, output_dir):
    """
    Cria e salva um modelo de ensemble usando os modelos treinados.

    Args:
        trained_models: Lista de modelos treinados
        model_paths: Lista de caminhos para os modelos treinados
        class_names: Lista de nomes das classes
        output_dir: Diretório para salvar o modelo de ensemble

    Returns:
        ensemble_path: Caminho para o modelo de ensemble salvo
    """
    ensemble_dir = os.path.join(output_dir, ENSEMBLE_DIR)
    os.makedirs(ensemble_dir, exist_ok=True)

    # Salvar informações do ensemble
    ensemble_info = {
        'model_paths': model_paths,
        'class_names': class_names,
        'num_models': len(trained_models)
    }

    ensemble_path = os.path.join(ensemble_dir, "ensemble_model.json")
    with open(ensemble_path, 'w') as f:
        json.dump(ensemble_info, f, indent=4)

    print(f"\n✅ Ensemble criado com {len(trained_models)} modelos e salvo em: {ensemble_path}")

    return ensemble_path


def train_and_evaluate_models(data_dir, output_dir, models_to_compare, epochs=30, batch_size=32):
    """
    Treina e avalia múltiplos modelos de SnakeCNN com transfer learning.
    Implementa técnicas avançadas para reduzir overfitting e melhorar generalização.
    """
    # Estrutura para armazenar resultados de todos os modelos
    results = {
        'models': {},
        'comparative': {
            'accuracy': {},
            'test_accuracy': {},  # Adicionado acurácia no conjunto de teste
            'training_time': {},
            'inference_time': {},
            'model_size_mb': {}
        }
    }

    # Cria diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Preparar dados com divisão em treino/validação/teste
    print(f"\n📊 Preparando dados de {data_dir}...")
    min_images_per_class = 20  # Aumentado o valor mínimo de 10 para 20

    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names, class_to_idx = prepare_data_with_test_set(
        data_dir, min_images_per_class
    )

    # Obter transformações avançadas
    train_transform, val_transform, test_transform = get_advanced_transforms()

    # Criar datasets
    train_dataset = SnakeDataset(train_paths, train_labels, train_transform)
    val_dataset = SnakeDataset(val_paths, val_labels, val_transform)
    test_dataset = SnakeDataset(test_paths, test_labels, test_transform)

    # Criar data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    num_classes = len(class_names)

    # Calcular pesos de classe para lidar com desbalanceamento
    class_count = Counter(train_labels)
    class_weights = torch.FloatTensor([
        max(class_count.values()) / count for count in [class_count[i] for i in range(num_classes)]
    ]).to(DEVICE)

    print("\n📊 Pesos de classe para lidar com desbalanceamento:")
    for i, weight in enumerate(class_weights):
        print(f"  - Classe {class_names[i]}: {weight.item():.2f}")

    # Lista para armazenar modelos treinados para ensemble
    trained_models = []
    model_paths = []

    # Iterar sobre cada variante de modelo
    for model_name in models_to_compare:
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        print(f"\n{'=' * 80}")
        print(f"🔥 Treinando modelo: {model_name}")
        print(f"{'=' * 80}")

        # Inicializa o modelo
        model = get_snake_model_with_transfer_learning(num_classes, base_model=model_name, freeze_backbone=True)
        model = model.to(DEVICE)

        # Configura otimizador com weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        # Critério com Label Smoothing para reduzir superconfiança
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        # Treinamento com descongelamento progressivo
        start_time = time.time()

        # Treinar com descongelamento progressivo
        history, best_model_state, best_val_acc = train_with_progressive_unfreezing(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=DEVICE,
            epochs=epochs,
            class_weights=class_weights,
            model_name=model_name
        )

        training_time = time.time() - start_time

        # Carregar o melhor modelo
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Adicionar à lista de modelos treinados para ensemble
        trained_models.append(model)

        # Medir o tempo de inferência
        model.eval()
        inference_times = []

        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(DEVICE)
                start = time.time()
                _ = model(inputs)
                end = time.time()
                inference_times.append((end - start) / inputs.size(0))  # Tempo por imagem

        avg_inference_time = sum(inference_times) / len(inference_times)

        # Medir o tamanho do modelo
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # MB

        # Avaliar no conjunto de teste
        test_acc, test_loss, y_true, y_pred = evaluate_on_test_set(model, test_loader, criterion, DEVICE)

        print(f"\n📊 Resultados no conjunto de teste para {model_name}:")
        print(f"  - Acurácia: {test_acc:.2f}%")
        print(f"  - Perda: {test_loss:.4f}")

        # Criar relatório de classificação
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

        # Criar matriz de confusão
        cm = confusion_matrix(y_true, y_pred)

        # Visualizar matriz de confusão
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Matriz de Confusão - {model_name}')
        plt.ylabel('Rótulo Verdadeiro')
        plt.xlabel('Rótulo Previsto')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(model_output_dir, 'confusion_matrix.png'))
        plt.close()

        # Salvar o modelo
        model_path = os.path.join(model_output_dir, f"snake_classifier_{model_name}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_names': class_names,
            'class_to_idx': class_to_idx,
            'history': history,
            'model_config': {
                'base_model': model_name,
                'num_classes': num_classes,
                'freeze_backbone': True
            }
        }, model_path)

        # Adicionar à lista de caminhos para modelos do ensemble
        model_paths.append(model_path)

        # Salvar o histórico e relatório
        history_path = os.path.join(model_output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)

        report_path = os.path.join(model_output_dir, "classification_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)

        # Gerar curvas de aprendizado
        generate_learning_curves(model_name, history, model_output_dir)

        # Armazenar resultados
        results['models'][model_name] = {
            'accuracy': best_val_acc,
            'test_accuracy': test_acc,
            'training_time_minutes': training_time / 60,
            'avg_inference_time_ms': avg_inference_time * 1000,  # Convertendo para ms
            'model_size_mb': model_size,
            'history': history,
            'classification_report': report,
            'model_path': model_path
        }

        results['comparative']['accuracy'][model_name] = best_val_acc
        results['comparative']['test_accuracy'][model_name] = test_acc
        results['comparative']['training_time'][model_name] = training_time / 60
        results['comparative']['inference_time'][model_name] = avg_inference_time * 1000
        results['comparative']['model_size_mb'][model_name] = model_size

        print(f"\n✅ Modelo {model_name} concluído:")
        print(f"   Melhor acurácia (validação): {best_val_acc:.2f}%")
        print(f"   Acurácia no teste: {test_acc:.2f}%")
        print(f"   Tempo de treinamento: {training_time / 60:.2f} minutos")
        print(f"   Tempo médio de inferência: {avg_inference_time * 1000:.2f} ms por imagem")
        print(f"   Tamanho do modelo: {model_size:.2f} MB")

    # Criar ensemble dos melhores modelos
    if len(trained_models) >= 2:
        create_ensemble_model(trained_models, model_paths, class_names, output_dir)

    # Gerar gráficos comparativos
    generate_comparative_charts(results['comparative'], output_dir)

    # Salvar resultados comparativos
    results_path = os.path.join(output_dir, "comparative_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    return results


def compare_all_snake_models(data_dir, output_dir, models_to_compare=None, epochs=30, batch_size=32):
    """
    Compara o desempenho de diferentes modelos pré-treinados para classificação de serpentes.
    """
    # Modelos para comparação
    if models_to_compare is None:
        models_to_compare = [
            'resnet50',
            'vgg16',  # Colocados primeiro porque tiveram melhor desempenho
            'efficientnet_b3',
            'densenet169',
            'inception_v3',
            'unet'
        ]

    # Treinar e avaliar todos os modelos
    results = train_and_evaluate_models(
        data_dir=data_dir,
        output_dir=output_dir,
        models_to_compare=models_to_compare,
        epochs=epochs,
        batch_size=batch_size
    )

    return results


def main():
    """
    Função principal para treinamento e comparação de modelos.
    """
    parser = argparse.ArgumentParser(description='Treinar e comparar múltiplos modelos de classificação de serpentes')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Diretório contendo as imagens originais organizadas por classe')
    parser.add_argument('--output-dir', type=str, default='resultados_comparacao',
                        help='Diretório base para salvar resultados dos modelos')
    parser.add_argument('--augmented-dir', type=str, default=r'C:\tcc\modelo\augmented',
                        help='Diretório fixo para dados aumentados')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Número de épocas para treinar cada modelo')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Tamanho do batch para treinamento')
    parser.add_argument('--include-models', type=str, nargs='+',
                        default=['resnet50', 'vgg16', 'efficientnet_b3', 'densenet169', 'inception_v3', 'unet'],
                        help='Modelos a incluir na comparação')
    parser.add_argument('--skip-augmentation', action='store_true',
                        help='Pular a etapa de aumentação de dados')
    parser.add_argument('--force-augmentation', action='store_true',
                        help='Forçar a geração de novos dados aumentados mesmo se já existirem')

    args = parser.parse_args()

    # Adicionar timestamp ao diretório de saída para não sobrescrever resultados anteriores
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"comparacao_{timestamp}")

    print("\n" + "=" * 80)
    print(f"🐍 INICIANDO COMPARAÇÃO DE MODELOS PARA CLASSIFICAÇÃO DE SERPENTES 🐍".center(80))
    print("=" * 80 + "\n")

    # Verificar se devemos usar dados aumentados existentes ou gerar novos
    augmented_dir = args.augmented_dir
    data_dir = args.data_dir

    if not args.skip_augmentation and not args.force_augmentation:
        print("\n🔍 Verificando se existem dados aumentados...")

        # Verificar se o diretório de dados aumentados já existe
        if os.path.exists(augmented_dir) and os.path.isdir(augmented_dir):
            # Verificar se há subdiretorios (classes) com imagens
            classes = [d for d in os.listdir(augmented_dir) if os.path.isdir(os.path.join(augmented_dir, d))]
            if classes:
                # Verificar se há imagens nas classes
                has_images = False
                total_images = 0
                valid_classes = 0

                for class_name in classes:
                    class_dir = os.path.join(augmented_dir, class_name)
                    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if images:
                        has_images = True
                        valid_classes += 1
                        total_images += len(images)

                if has_images:
                    print(f"✅ Encontrados dados aumentados válidos em: {augmented_dir}")
                    print(f"   - {valid_classes} classes com um total de {total_images} imagens")
                    data_dir = augmented_dir
                else:
                    print("❌ Diretório de dados aumentados existe, mas não contém imagens válidas.")
                    print("🔄 Preparando geração de novos dados aumentados...")
                    os.makedirs(augmented_dir, exist_ok=True)
                    augment_and_save_images(args.data_dir, augmented_dir, TARGET_IMAGES_PER_CLASS)
                    data_dir = augmented_dir
            else:
                print("❌ Diretório de dados aumentados existe, mas não contém classes.")
                print("🔄 Preparando geração de novos dados aumentados...")
                os.makedirs(augmented_dir, exist_ok=True)
                augment_and_save_images(args.data_dir, augmented_dir, TARGET_IMAGES_PER_CLASS)
                data_dir = augmented_dir
        else:
            print("❌ Diretório de dados aumentados não existe.")
            print("🔄 Preparando geração de novos dados aumentados...")
            os.makedirs(augmented_dir, exist_ok=True)
            augment_and_save_images(args.data_dir, augmented_dir, TARGET_IMAGES_PER_CLASS)
            data_dir = augmented_dir
    elif args.force_augmentation:
        print("\n🔄 Forçando a geração de novos dados aumentados...")
        if os.path.exists(augmented_dir):
            print(f"   - Limpando diretório existente: {augmented_dir}")
            # Opção 1: Remover completamente o diretório (mais seguro se tiver permissão)
            # shutil.rmtree(augmented_dir)
            # os.makedirs(augmented_dir)

            # Opção 2: Limpar o conteúdo mas manter o diretório (preferível se tiver problema de permissão)
            for class_dir in os.listdir(augmented_dir):
                class_path = os.path.join(augmented_dir, class_dir)
                if os.path.isdir(class_path):
                    for file in os.listdir(class_path):
                        file_path = os.path.join(class_path, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
        else:
            os.makedirs(augmented_dir, exist_ok=True)

        augment_and_save_images(args.data_dir, augmented_dir, TARGET_IMAGES_PER_CLASS)
        data_dir = augmented_dir
    else:
        # Se skip_augmentation for True, usamos o diretório de dados original
        print("\n⏩ Pulando verificação/geração de dados aumentados conforme solicitado.")
        data_dir = args.data_dir

    # Criar diretório de saída para os resultados do treinamento
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📋 Configurações para treinamento:")
    print(f"  - Diretório de dados: {data_dir}")
    print(f"  - Diretório de saída: {output_dir}")
    print(f"  - Épocas por modelo: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Modelos incluídos: {', '.join(args.include_models)}")

    # Treinar e comparar todos os modelos
    results = compare_all_snake_models(
        data_dir=data_dir,
        output_dir=output_dir,
        models_to_compare=args.include_models,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Exibir resumo final
    print("\n" + "=" * 80)
    print("🏆 RESUMO FINAL DA COMPARAÇÃO 🏆".center(80))
    print("=" * 80 + "\n")

    # Ordenar modelos do melhor para o pior em acurácia de teste
    sorted_models = sorted(
        results['comparative']['test_accuracy'].items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Mostrar ranking
    print("📊 RANKING DE MODELOS (Por acurácia de teste):")
    for i, (model, accuracy) in enumerate(sorted_models):
        val_accuracy = results['comparative']['accuracy'][model]
        print(f"{i + 1}. {model}")
        print(f"   ├─ Acurácia (teste): {accuracy:.2f}%")
        print(f"   ├─ Acurácia (validação): {val_accuracy:.2f}%")
        print(f"   ├─ Tempo treinamento: {results['comparative']['training_time'][model]:.2f} minutos")
        print(f"   ├─ Tempo inferência: {results['comparative']['inference_time'][model]:.2f} ms/imagem")
        print(f"   └─ Tamanho: {results['comparative']['model_size_mb'][model]:.2f} MB")

    print(f"\n📂 Todos os resultados, modelos treinados e gráficos foram salvos em: {output_dir}")
    print("\n✨ Comparação concluída com sucesso! ✨")

    print("\n📄 Use o script de inferência para testar o ensemble ou os modelos individuais.")


if __name__ == "__main__":
    main()