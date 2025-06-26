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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split, KFold
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, \
    average_precision_score

"""
Script de treinamento otimizado para classificação binária de serpentes.
Configurado com valores padrão para tratamento de dataset desbalanceado,
com mais amostras de cobras não peçonhentas que peçonhentas.

Este script inclui por padrão:
- Peso 2x maior para a classe peçonhenta durante o treinamento
- Augmentation extra para a classe de cobras peçonhentas (50% mais imagens)
- Transformações de data augmentation agressivas
- Técnicas para evitar overfitting (Label Smoothing, Dropout)
- Mixup e CutMix para melhorar generalização
- K-Fold Cross Validation para avaliação robusta
- Otimização de threshold para priorizar a detecção de cobras peçonhentas
- Visualizações detalhadas para análise de desempenho

Uso básico:
    python snake_training.py --data-dir C:/caminho/para/imagens

Parâmetros opcionais:
    --output-dir: Diretório para salvar os resultados (padrão: resultados_snake)
    --model: Modelo base a ser usado (padrão: resnet50)
    --epochs: Número de épocas para treinar (padrão: 30)
    --kfold: Número de folds para validação cruzada (padrão: 0, desabilitado)
    --label-smoothing: Valor para suavização de rótulos (padrão: 0.1)
    --use-mixup-cutmix: Ativar técnicas de Mixup e CutMix (padrão: True)
"""

# Configurações globais (já otimizadas)
BATCH_SIZE = 32  # Tamanho do lote para treinamento
NUM_EPOCHS = 60  # Número total de épocas para treinamento
LEARNING_RATE = 0.01  # Taxa de aprendizado inicial
IMAGE_SIZE = 224  # Tamanho das imagens após redimensionamento
NUM_WORKERS = 8  # Número de workers para carregamento de dados paralelo (era 4 antes)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Usa GPU se disponível
TARGET_IMAGES_PER_CLASS = 800  # Número alvo de imagens por classe após aumentação
WEIGHT_DECAY = 3e-4  # Regularização L2 para evitar overfitting
VENOMOUS_WEIGHT = 1.0  # Peso para classe peçonhenta (balanceamento)
DROPOUT_RATE = 0.5  # Taxa de dropout para evitar overfitting
EXTRA_VENOMOUS_AUGMENTATION = False  # Ativa aumentação extra para cobras peçonhentas
STRONG_AUGMENTATION = True  # Ativa transformações agressivas de aumentação
USE_MIXUP_CUTMIX = False  # Ativa técnicas Mixup e CutMix para melhorar generalização
EARLY_STOPING_PATIENCE = 10  # Número de épocas sem melhoria para parar o treinamento

# Diretórios para salvar resultados
MODELS_DIR = "models"  # Diretório para salvar os modelos treinados
PLOTS_DIR = "plots"  # Diretório para salvar as visualizações
REPORTS_DIR = "reports"  # Diretório para salvar os relatórios de desempenho
FOLDS_DIR = "folds"  # Diretório para salvar modelos de cada fold


class MixupCutmixAugmentation:
    """
    Implementação de Mixup e CutMix para melhorar a generalização do modelo.

    Estas técnicas combinam imagens e rótulos diferentes durante o treinamento,
    fazendo com que o modelo aprenda fronteiras de decisão mais suaves e se
    torne menos suscetível a overconfidence e overfitting.
    """

    def __init__(self, mixup_alpha=1.0, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5,
                 label_smoothing=0.1, num_classes=2):
        """
        Inicializa os parâmetros para Mixup e CutMix.

        Args:
            mixup_alpha: Parâmetro alpha para a distribuição Beta no Mixup
            cutmix_alpha: Parâmetro alpha para a distribuição Beta no CutMix
            prob: Probabilidade de aplicar mixup ou cutmix
            switch_prob: Probabilidade de mudar de mixup para cutmix
            label_smoothing: Valor de suavização para os rótulos one-hot
            num_classes: Número de classes no dataset
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def to_one_hot(self, targets, num_classes=2):
        """
        Converte rótulos de classe para formato one-hot.

        Args:
            targets: Tensor com os índices das classes
            num_classes: Número total de classes

        Returns:
            Tensor com representação one-hot dos rótulos
        """
        targets_one_hot = torch.zeros(targets.size(0), num_classes, device=targets.device)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        # Aplicar label smoothing se necessário
        if self.label_smoothing > 0:
            targets_one_hot = (1 - self.label_smoothing) * targets_one_hot + \
                              self.label_smoothing / self.num_classes

        return targets_one_hot

    def mixup_data(self, x, y):
        """
        Implementa a técnica de mixup.

        Combina pares de imagens e seus rótulos linearmente com um peso lambda
        sorteado de uma distribuição Beta.

        Args:
            x: Batch de imagens
            y: Batch de rótulos

        Returns:
            mixed_x: Imagens misturadas
            y_a: Rótulos originais
            y_b: Rótulos permutados
            lam: Fator de mistura (lambda)
        """
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def cutmix_data(self, x, y):
        """
        Implementa a técnica de cutmix.

        Substitui uma região retangular aleatória de cada imagem
        com a mesma região de outra imagem do batch.

        Args:
            x: Batch de imagens
            y: Batch de rótulos

        Returns:
            mixed_x: Imagens com regiões trocadas
            y_a: Rótulos originais
            y_b: Rótulos permutados
            lam: Fator de mistura ajustado pela área de corte
        """
        if self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        y_a, y_b = y, y[index]

        # Tamanho da imagem
        input_size = x.size(2)

        # Determinar o tamanho do corte baseado em lambda
        cut_ratio = np.sqrt(1. - lam)
        cut_w = int(input_size * cut_ratio)
        cut_h = int(input_size * cut_ratio)

        # Centro aleatório do corte
        cx = np.random.randint(input_size)
        cy = np.random.randint(input_size)

        # Limites do corte
        bbx1 = np.clip(cx - cut_w // 2, 0, input_size)
        bby1 = np.clip(cy - cut_h // 2, 0, input_size)
        bbx2 = np.clip(cx + cut_w // 2, 0, input_size)
        bby2 = np.clip(cy + cut_h // 2, 0, input_size)

        # Realizar o CutMix
        mixed_x = x.clone()
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

        # Ajustar lambda baseado na área real do corte
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input_size * input_size))

        return mixed_x, y_a, y_b, lam

    def __call__(self, x, y):
        """
        Aplica Mixup ou CutMix aleatoriamente.

        Args:
            x: Batch de imagens
            y: Batch de rótulos

        Returns:
            mixed_x: Batch de imagens após Mixup/CutMix
            mixed_y_a: Rótulos originais
            mixed_y_b: Rótulos permutados
            lam: Fator de mistura
        """
        if random.random() < self.prob:
            if random.random() < self.switch_prob:
                return self.mixup_data(x, y)
            else:
                return self.cutmix_data(x, y)
        return x, y, y, 1.0


# Função de perda para Mixup/CutMix
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calcula a perda para saídas de Mixup/CutMix.

    Combina as perdas para os dois conjuntos de rótulos
    usando o fator de mistura lambda.

    Args:
        criterion: Função de perda (ex: CrossEntropyLoss)
        pred: Saída do modelo
        y_a: Rótulos originais
        y_b: Rótulos permutados
        lam: Fator de mistura

    Returns:
        Perda calculada com ponderação lambda
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingLoss(nn.Module):
    """
    Implementação de Label Smoothing para melhorar calibração e generalização.

    Label smoothing suaviza os rótulos one-hot (ex: [0,1] -> [0.05, 0.95]),
    o que ajuda a prevenir overfitting e melhora a calibração do modelo.
    """

    def __init__(self, classes=2, smoothing=0.1, weight=None, reduction='mean'):
        """
        Inicializa os parâmetros para Label Smoothing.

        Args:
            classes: Número de classes no problema
            smoothing: Fator de suavização (entre 0 e 1)
            weight: Pesos opcionais para cada classe
            reduction: Tipo de redução ('mean', 'sum' ou 'none')
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Calcula a perda com label smoothing.

        Args:
            pred: Saídas do modelo (logits)
            target: Rótulos verdadeiros

        Returns:
            Perda calculada com label smoothing
        """
        pred = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        if self.weight is not None:
            weight_exp = self.weight.unsqueeze(0).expand_as(pred)
            loss = torch.sum(-true_dist * pred * weight_exp, dim=1)
        else:
            loss = torch.sum(-true_dist * pred, dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SnakeDataset(Dataset):
    """
    Dataset personalizado para as imagens de serpentes.

    Implementa a interface do PyTorch Dataset para carregamento
    eficiente de imagens de serpentes e seus rótulos.
    """

    def __init__(self, image_paths, labels, transform=None):
        """
        Inicializa o dataset com caminhos de imagens e rótulos.

        Args:
            image_paths: Lista com caminhos para arquivos de imagem
            labels: Lista com rótulos correspondentes (0=não peçonhenta, 1=peçonhenta)
            transform: Transformações a serem aplicadas nas imagens
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Retorna o número de imagens no dataset"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Carrega uma imagem e seu rótulo pelo índice.

        Args:
            idx: Índice da imagem a ser carregada

        Returns:
            Tupla (imagem, rótulo)
        """
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print(f"⚠️ Erro ao carregar imagem {self.image_paths[idx]}: {e}")
            # Retornar uma imagem vazia como fallback
            dummy_image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            return dummy_image, self.labels[idx]


class SnakeBinaryClassifier(nn.Module):
    """
    Modelo de classificação binária para serpentes.

    Utiliza uma arquitetura de redes neurais convolucionais pré-treinadas
    como backbone, seguida por camadas classificadoras customizadas.
    """

    def __init__(self, base_model='resnet50', freeze_backbone=True, dropout_rate=0.5):
        """
        Inicializa o modelo com o backbone escolhido.

        Args:
            base_model: Modelo base a ser usado (resnet50, efficientnet_b3, etc.)
            freeze_backbone: Se True, congela os pesos do backbone
            dropout_rate: Taxa de dropout para as camadas classificadoras
        """
        super(SnakeBinaryClassifier, self).__init__()

        self.base_model_name = base_model
        self.freeze_backbone = freeze_backbone

        # Seleção do backbone conforme o modelo base escolhido
        if base_model == 'resnet50':
            base = models.resnet50(weights='IMAGENET1K_V1')
            backbone = nn.Sequential(*list(base.children())[:-2])
            feature_size = 2048
        elif base_model == 'efficientnet_b3':
            base = models.efficientnet_b3(weights='IMAGENET1K_V1')
            backbone = base.features
            feature_size = 1536
        elif base_model == 'efficientnet_b0':
            base = models.efficientnet_b0(weights='IMAGENET1K_V1')
            backbone = base.features
            feature_size = 1280
        elif base_model == 'densenet169':
            base = models.densenet169(weights='IMAGENET1K_V1')
            backbone = base.features
            feature_size = 1664
        elif base_model == 'vgg16':
            base = models.vgg16(weights='IMAGENET1K_V1')
            backbone = base.features
            feature_size = 512
        else:
            raise ValueError(f"Modelo base não suportado: {base_model}")

        # Congela os pesos do backbone se solicitado
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        self.backbone = backbone
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(128, 2)
        )

        # Inicialização aprimorada dos pesos
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Propaga a entrada através do modelo.

        Args:
            x: Tensor de entrada (batch de imagens)

        Returns:
            Logits para as duas classes (não peçonhenta, peçonhenta)
        """
        # Extrai features pelo backbone
        features = self.backbone(x)
        features = self.adaptive_pool(features)
        # Passa pelas camadas classificadoras
        x = self.classifier(features)
        return x

    def unfreeze_layers(self, percentage=0.3):
        """
        Descongela uma porcentagem das últimas camadas do backbone.

        Útil para fine-tuning gradual durante o treinamento.

        Args:
            percentage: Porcentagem do backbone a ser descongelada (do fim para o início)
        """
        if not self.freeze_backbone:
            return

        if hasattr(self.backbone, "children"):
            layers = list(self.backbone.children())
            num_layers = len(layers)
            start_idx = int((1 - percentage) * num_layers)
            for i, layer in enumerate(layers):
                if i >= start_idx:
                    for param in layer.parameters():
                        param.requires_grad = True

            print(f"🔓 Descongeladas {num_layers - start_idx} de {num_layers} camadas do backbone")


def create_output_directories(output_base_dir, with_folds=False):
    """
    Cria os diretórios necessários para salvar os resultados.

    Args:
        output_base_dir: Diretório base para salvar resultados
        with_folds: Se True, cria também diretório para folds

    Returns:
        Dicionário com os caminhos dos diretórios criados
    """
    base_path = Path(output_base_dir)

    directories = {
        'base': str(base_path),
        'models': str(base_path / MODELS_DIR),
        'plots': str(base_path / PLOTS_DIR),
        'reports': str(base_path / REPORTS_DIR)
    }

    if with_folds:
        directories['folds'] = str(base_path / FOLDS_DIR)

    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Diretório criado/verificado: {dir_path}")

    return directories


def augment_and_save_images(data_dir, output_dir, target_images_per_class, extra_venomous=EXTRA_VENOMOUS_AUGMENTATION):
    """
    Gera imagens aumentadas para cada classe até atingir o número alvo por classe.

    Aplica diversas transformações aleatórias nas imagens originais para
    aumentar o tamanho do dataset e melhorar a generalização do modelo.

    Args:
        data_dir: Diretório com as imagens originais
        output_dir: Diretório para salvar as imagens aumentadas
        target_images_per_class: Número alvo de imagens por classe após aumentação
        extra_venomous: Se True, gera mais imagens para a classe peçonhenta

    Returns:
        Caminho para o diretório com as imagens aumentadas
    """
    print("\n" + "=" * 80)
    print("🔄 AUMENTAÇÃO DE DADOS - GERANDO IMAGENS SINTÉTICAS 🔄".center(80))
    print("=" * 80)

    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Transformações regulares para data augmentation
    regular_transforms = [
        transforms.RandomHorizontalFlip(p=0.7),
        transforms.RandomVerticalFlip(p=0.7),
        transforms.RandomRotation(60),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomAffine(degrees=30, translate=(0.3, 0.3), scale=(0.7, 1.3)),
        transforms.RandomPerspective(distortion_scale=0.6, p=0.7),
        transforms.GaussianBlur(kernel_size=5),
        transforms.RandomGrayscale(p=0.1),
    ]

    # Transformações adicionais para aumentar a diversidade das cobras peçonhentas
    extra_transforms = [
        transforms.RandomAffine(degrees=45, translate=(0.4, 0.4), scale=(0.6, 1.4), shear=15),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
        transforms.RandomPerspective(distortion_scale=0.7, p=0.8),
        transforms.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.7, 1.0), ratio=(0.7, 1.3)),
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

        # Definir alvo específico para a classe (mais imagens para peçonhentas)
        current_target = target_images_per_class
        is_venomous = class_name.lower() == "peconhentas"
        if is_venomous and extra_venomous:
            current_target = int(target_images_per_class * 1.5)  # 50% mais imagens para peçonhentas
            print(f"⚠️ Aumentando alvo para classe '{class_name}' para {current_target} imagens")

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
        if len(original_images) >= current_target:
            print(f"  ✅ Já possui {len(original_images)} imagens (>= {current_target}). Pulando aumentação.")
            class_stats.append((class_name, len(original_images), len(original_images), 0))
            continue

        # Calcular quantas imagens aumentadas precisamos gerar por imagem original
        num_augmentations_per_image = (current_target - len(original_images)) // len(original_images)
        remaining_augmentations = (current_target - len(original_images)) % len(original_images)

        print(f"  ↳ Gerando ~{num_augmentations_per_image} variações por imagem original")

        # Usar transformações especiais para cobras peçonhentas
        transforms_to_use = regular_transforms
        if is_venomous and extra_venomous:
            transforms_to_use = regular_transforms + extra_transforms
            print(f"  ↳ Usando conjunto estendido de transformações para cobras peçonhentas")

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

                # Aplicar transformações aleatórias
                random.shuffle(transforms_to_use)
                # Mais transformações por imagem para cobras peçonhentas
                num_transforms = random.randint(3, min(6 if is_venomous else 4, len(transforms_to_use)))

                for t in range(num_transforms):
                    try:
                        aug_img = transforms_to_use[t](aug_img)
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
    Prepara os dados, dividindo em conjuntos de treino, validação e teste.

    Faz a divisão estratificada dos dados garantindo que as proporções
    de classes sejam mantidas nos três conjuntos.

    Args:
        data_dir: Diretório com as imagens
        min_images_per_class: Número mínimo de imagens para considerar uma classe

    Returns:
        Caminhos e rótulos para os conjuntos de treino, validação e teste,
        além dos nomes das classes e mapeamento para índices
    """
    image_paths = []
    labels = []
    class_names = []
    class_counts = {}
    valid_class_dirs = []

    # Mapeamento fixo das classes:
    # 0 - naopeconhentas (não peçonhentas)
    # 1 - peconhentas (peçonhentas)
    class_mapping = {
        "naopeconhentas": 0,
        "peconhentas": 1
    }

    # Primeiro, conta as imagens em cada classe
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
    class_to_idx = class_mapping  # Usando o mapeamento fixo

    print("\nClasses incluídas no treinamento:")
    for class_name, class_dir in valid_class_dirs:
        class_names.append(class_name)

        # Adiciona as imagens e seus rótulos
        class_image_paths = [os.path.join(class_dir, img_name)
                             for img_name in os.listdir(class_dir)
                             if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))]

        image_paths.extend(class_image_paths)
        labels.extend([class_to_idx[class_name]] * len(class_image_paths))

        print(f"Classe {class_name} (índice {class_to_idx[class_name]}): {len(class_image_paths)} imagens")

    print(f"\nTotal de classes válidas: {len(class_names)}")
    print(f"Total de imagens: {len(image_paths)}")

    # Se não houver classes válidas, levanta um erro
    if len(class_names) == 0:
        raise ValueError(f"Nenhuma classe encontrada com pelo menos {min_images_per_class} imagens!")

    # Garantir estratificação mesmo com poucas classes (apenas 2)
    stratify = labels if len(set(labels)) > 1 else None

    # Dividir em treino, validação e teste (60%, 20%, 20%)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.4, random_state=42, stratify=stratify
    )

    # Dividir os dados restantes em validação e teste
    stratify_temp = temp_labels if len(set(temp_labels)) > 1 else None
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=stratify_temp
    )

    print(f"\nDivisão dos dados:")
    print(f"  - Treinamento: {len(train_paths)} imagens")
    print(f"  - Validação: {len(val_paths)} imagens")
    print(f"  - Teste: {len(test_paths)} imagens")

    # Confirmando distribuição das classes
    train_class_dist = Counter(train_labels)
    val_class_dist = Counter(val_labels)
    test_class_dist = Counter(test_labels)

    print("\nDistribuição de classes:")
    for i, class_name in enumerate(class_names):
        class_idx = class_to_idx[class_name]
        print(f"  - {class_name} (índice {class_idx}):")
        print(f"      Treino: {train_class_dist.get(class_idx, 0)} imagens")
        print(f"      Validação: {val_class_dist.get(class_idx, 0)} imagens")
        print(f"      Teste: {test_class_dist.get(class_idx, 0)} imagens")

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names, class_to_idx


def get_advanced_transforms(use_strong_augmentation=STRONG_AUGMENTATION):
    """
    Retorna transformações para treinamento, validação e teste.

    Define conjuntos de transformações específicas para cada fase:
    - Treino: com augmentations para aumentar a robustez
    - Validação/Teste: apenas normalização e redimensionamento

    Args:
        use_strong_augmentation: Se True, usa transformações mais agressivas

    Returns:
        Tripla com transformações para treino, validação e teste
    """
    # Transformações de treino com data augmentation
    if use_strong_augmentation:
        # Transformações mais agressivas para aumentar a generalização
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.8),
            transforms.RandomVerticalFlip(p=0.6),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
            transforms.RandomAffine(degrees=40, translate=(0.3, 0.3), scale=(0.7, 1.3), shear=20),
            transforms.RandomPerspective(distortion_scale=0.6, p=0.7),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomEqualize(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
        ])
    else:
        # Transformações padrão
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(60),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Transformações para validação (apenas redimensionamento e normalização)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transformações para teste (idênticas às de validação)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform, test_transform


def generate_learning_curves(model_name, history, output_dir):
    """
    Gera e salva gráficos de curvas de aprendizado para um modelo.

    Plota a evolução da acurácia e da perda durante o treinamento,
    comparando os conjuntos de treino e validação.

    Args:
        model_name: Nome do modelo para incluir no título
        history: Dicionário com histórico de treino e validação
        output_dir: Diretório para salvar as curvas
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


def plot_roc_curve(model, test_loader, device, class_names, output_dir):
    """
    Gera a curva ROC e a curva Precision-Recall para o modelo.

    A curva ROC (Receiver Operating Characteristic) mostra o desempenho
    de classificação em diferentes limiares, plotando a taxa de verdadeiros
    positivos contra a taxa de falsos positivos.

    Args:
        model: Modelo treinado
        test_loader: DataLoader com dados de teste
        device: Dispositivo (CPU/GPU)
        class_names: Nomes das classes
        output_dir: Diretório para salvar as curvas

    Returns:
        AUC-ROC e Average Precision calculados
    """
    model.eval()
    y_true = []
    y_scores = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Para classificação binária, usamos a probabilidade da classe positiva (classe 1)
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probabilities[:, 1].cpu().numpy())  # Prob. da classe positiva (peçonhenta)

    # Calcula a curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plota a curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC - Classificação Binária de Serpentes')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Adicionar pontos de threshold importantes
    thresholds_to_show = [0.1, 0.3, 0.5, 0.7, 0.9]
    for threshold in thresholds_to_show:
        # Encontrar o índice mais próximo do threshold desejado
        idx = np.argmin(np.abs(thresholds - threshold))
        plt.plot(fpr[idx], tpr[idx], 'ro')
        plt.annotate(f'T={threshold:.1f}',
                     (fpr[idx], tpr[idx]),
                     textcoords="offset points",
                     xytext=(10, 5),
                     ha='center')

    # Salva a figura
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    # Gerar curva Precision-Recall (importante para datasets desbalanceados)
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'AP = {avg_precision:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall - Classificação Binária de Serpentes')
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    # Adicionar linha de base para comparação
    no_skill = len([x for x in y_true if x == 1]) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='red',
             label='No Skill')

    # Adicionar pontos para thresholds importantes
    thresholds_pr = np.append(thresholds_pr, 1.0)  # precision_recall_curve retorna um threshold a menos
    for threshold in thresholds_to_show:
        if threshold >= min(thresholds_pr) and threshold <= max(thresholds_pr):
            idx = np.argmin(np.abs(thresholds_pr - threshold))
            plt.plot(recall[idx], precision[idx], 'go')
            plt.annotate(f'T={threshold:.1f}',
                         (recall[idx], precision[idx]),
                         textcoords="offset points",
                         xytext=(10, 5),
                         ha='center')

    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()

    return roc_auc, avg_precision


def find_optimal_threshold(y_true, y_scores, output_dir):
    """
    Encontra o threshold ótimo para classificação baseado em diferentes critérios.

    Analisa diversos critérios para escolha do melhor limiar de decisão:
    - Índice de Youden (máxima diferença entre TPR e FPR)
    - Distância mínima ao ponto (0,1)
    - F1-score máximo
    - Alta recall (para priorizar detecção de serpentes peçonhentas)

    Args:
        y_true: Rótulos verdadeiros
        y_scores: Pontuações (probabilidades) do modelo
        output_dir: Diretório para salvar análises

    Returns:
        Dicionário com thresholds ótimos para cada critério
    """
    # Calcular a curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Correção importante: fpr e tpr têm um elemento a mais que thresholds
    # Vamos garantir que todos tenham o mesmo tamanho
    if len(fpr) > len(thresholds):
        fpr = fpr[:-1]
        tpr = tpr[:-1]

    # Critério de Youden's J (maximiza TPR - FPR)
    j_scores = tpr - fpr
    optimal_idx_j = np.argmax(j_scores)
    optimal_threshold_j = thresholds[optimal_idx_j]

    # Critério de distância mínima ao ponto (0,1) - ponto ideal
    distances = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
    optimal_idx_d = np.argmin(distances)
    optimal_threshold_d = thresholds[optimal_idx_d]

    # Critério de F1-score (harmônica entre precisão e recall)
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    # Precision e recall têm um elemento a mais que thresholds_pr
    if len(precision) > len(thresholds_pr) + 1:
        precision = precision[:-1]
        recall = recall[:-1]

    # Adicionar o threshold 1.0 para completar
    thresholds_pr = np.append(thresholds_pr, 1.0)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # evitar divisão por zero
    optimal_idx_f1 = np.argmax(f1_scores)
    optimal_threshold_f1 = thresholds_pr[optimal_idx_f1]

    # Plotar thresholds vs métricas
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(thresholds, tpr, 'b-', label='Taxa de Verdadeiros Positivos')
    plt.plot(thresholds, fpr, 'r-', label='Taxa de Falsos Positivos')
    plt.axvline(x=optimal_threshold_j, color='g', linestyle='--',
                label=f'Threshold J: {optimal_threshold_j:.3f}')
    plt.axvline(x=optimal_threshold_d, color='m', linestyle='--',
                label=f'Threshold D: {optimal_threshold_d:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Taxa')
    plt.title('TPR e FPR vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    # Garantir que todos os arrays têm o mesmo tamanho
    min_len = min(len(thresholds_pr), len(precision), len(recall), len(f1_scores))
    thresholds_plot = thresholds_pr[:min_len]
    precision_plot = precision[:min_len]
    recall_plot = recall[:min_len]
    f1_scores_plot = f1_scores[:min_len]

    plt.plot(thresholds_plot, precision_plot, 'b-', label='Precisão')
    plt.plot(thresholds_plot, recall_plot, 'r-', label='Recall')
    plt.plot(thresholds_plot, f1_scores_plot, 'g-', label='F1 Score')
    plt.axvline(x=optimal_threshold_f1, color='c', linestyle='--',
                label=f'Threshold F1: {optimal_threshold_f1:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precisão, Recall e F1 vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'))
    plt.close()

    # Especialmente para segurança com cobras peçonhentas, queremos priorizar recall (TPR)
    # Vamos calcular um threshold que dê pelo menos 95% de recall
    high_recall_idx = np.where(recall >= 0.95)[0]
    if len(high_recall_idx) > 0:
        high_recall_idx = high_recall_idx[-1]  # último índice com recall >= 95%
        high_recall_threshold = thresholds_pr[high_recall_idx] if high_recall_idx < len(thresholds_pr) else 0.3
    else:
        # Se não conseguir 95%, use o valor que dá o maior recall
        high_recall_idx = np.argmax(recall)
        high_recall_threshold = thresholds_pr[high_recall_idx] if high_recall_idx < len(thresholds_pr) else 0.3

    return {
        'youden_j': float(optimal_threshold_j),  # Garantir que seja serializável
        'min_distance': float(optimal_threshold_d),
        'max_f1': float(optimal_threshold_f1),
        'high_recall': float(high_recall_threshold)
    }


def train_with_progressive_unfreezing(model, criterion, optimizer, train_loader, val_loader, device, epochs,
                                      class_names, output_dir, use_mixup_cutmix=USE_MIXUP_CUTMIX):
    """
    Treina o modelo com descongelamento gradual do backbone e uso de Mixup/CutMix.

    Implementa uma estratégia avançada de treinamento onde camadas do backbone são
    gradualmente descongeladas ao longo do treinamento, permitindo um fine-tuning
    progressivo das camadas pré-treinadas.

    Args:
        model: Modelo a ser treinado
        criterion: Função de perda
        optimizer: Otimizador
        train_loader: DataLoader para dados de treino
        val_loader: DataLoader para dados de validação
        device: Dispositivo (CPU/GPU)
        epochs: Número de épocas
        class_names: Nomes das classes
        output_dir: Diretório para salvar resultados
        use_mixup_cutmix: Se True, aplica técnicas de Mixup/CutMix

    Returns:
        Modelo treinado, histórico de treinamento e melhor acurácia de validação
    """
    # Inicializar histórico
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    # Rastreamento do melhor modelo
    best_val_acc = 0.0
    best_model_state = None
    best_val_loss = float('inf')
    best_model_state_loss = None

    # Usar OneCycleLR desde o início
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=len(train_loader) * epochs,
        pct_start=0.3,
        div_factor=10.0,
        final_div_factor=100.0
    )

    # Early stopping com paciência aumentada
    patience = 20  # Aumentado de 12
    counter = 0

    # Definir estágios mais graduais de descongelamento
    unfreeze_stage1 = epochs // 6  # Descongela apenas 10% primeiro
    unfreeze_stage2 = epochs // 3  # Depois 30%
    unfreeze_stage3 = epochs // 2  # Depois 70%
    unfreeze_stage4 = int(epochs * 0.7)  # Depois 100% mas muito mais tarde

    # Inicializar o Mixup/CutMix com parâmetros ajustados
    mixup_cutmix = None
    if use_mixup_cutmix:
        mixup_cutmix = MixupCutmixAugmentation(mixup_alpha=0.8, cutmix_alpha=0.8, prob=0.5,
                                               switch_prob=0.3, label_smoothing=0.1, num_classes=len(class_names))

    print(f"\n{'=' * 80}")
    print(f"🔥 Treinando modelo para classificação binária de serpentes")
    print(f"{'=' * 80}")
    print(f"📋 Plano de treinamento:")
    print(f"   - Épocas 1-{unfreeze_stage1}: Backbone congelado")
    print(f"   - Épocas {unfreeze_stage1 + 1}-{unfreeze_stage2}: 10% do backbone descongelado")
    print(f"   - Épocas {unfreeze_stage2 + 1}-{unfreeze_stage3}: 30% do backbone descongelado")
    print(f"   - Épocas {unfreeze_stage3 + 1}-{unfreeze_stage4}: 70% do backbone descongelado")
    print(f"   - Épocas {unfreeze_stage4 + 1}-{epochs}: Backbone completamente descongelado")
    if use_mixup_cutmix:
        print(f"   - Usando Mixup/CutMix com probabilidade 0.5")

    for epoch in range(epochs):
        # Atualizar estado de congelamento das camadas com base no progresso
        if epoch == unfreeze_stage1:
            print(f"\n🔓 Descongelando 10% do backbone (época {epoch + 1})...")
            model.unfreeze_layers(0.1)  # Descongela apenas 10%

            # Criar um novo otimizador com taxas de aprendizado específicas
            params = [
                {'params': [p for n, p in model.backbone.named_parameters() if p.requires_grad],
                 'lr': LEARNING_RATE * 0.02},  # Taxa muito baixa para camadas recém-descongeladas
                {'params': model.classifier.parameters(), 'lr': LEARNING_RATE * 0.8}
            ]
            optimizer = optim.AdamW(params, lr=LEARNING_RATE * 0.8, weight_decay=WEIGHT_DECAY)

            # Reiniciar o scheduler com o novo otimizador
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[LEARNING_RATE * 0.02, LEARNING_RATE * 0.8],
                total_steps=len(train_loader) * (epochs - epoch),
                pct_start=0.3,
                div_factor=10.0,
                final_div_factor=100.0
            )
            print(f"   - Otimizador e scheduler reiniciados com taxas de aprendizado ajustadas")

        elif epoch == unfreeze_stage2:
            print(f"\n🔓 Descongelando 30% do backbone (época {epoch + 1})...")
            model.unfreeze_layers(0.3)  # Descongela 30%

            # Criar um novo otimizador com taxas de aprendizado específicas
            params = [
                {'params': [p for n, p in model.backbone.named_parameters() if "layer4" in n and p.requires_grad],
                 'lr': LEARNING_RATE * 0.05},  # Taxa um pouco maior para últimas camadas
                {'params': [p for n, p in model.backbone.named_parameters() if "layer3" in n and p.requires_grad],
                 'lr': LEARNING_RATE * 0.02},  # Taxa menor para camadas anteriores
                {'params': [p for n, p in model.backbone.named_parameters()
                            if not any(x in n for x in ["layer3", "layer4"]) and p.requires_grad],
                 'lr': LEARNING_RATE * 0.01},  # Taxa ainda menor para camadas iniciais
                {'params': model.classifier.parameters(), 'lr': LEARNING_RATE * 0.5}
            ]
            optimizer = optim.AdamW(params, lr=LEARNING_RATE * 0.5, weight_decay=WEIGHT_DECAY * 0.8)

            # Aplicar clipping de gradiente
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Reiniciar scheduler
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[LEARNING_RATE * 0.05, LEARNING_RATE * 0.02, LEARNING_RATE * 0.01, LEARNING_RATE * 0.5],
                total_steps=len(train_loader) * (epochs - epoch),
                pct_start=0.3,
                div_factor=10.0,
                final_div_factor=100.0
            )
            print(f"   - Otimizador e scheduler reiniciados com taxas de aprendizado ajustadas")

        elif epoch == unfreeze_stage3:
            print(f"\n🔓 Descongelando 70% do backbone (época {epoch + 1})...")
            model.unfreeze_layers(0.7)  # Descongela 70%

            # Criar um novo otimizador com taxas de aprendizado específicas
            params = [
                {'params': [p for n, p in model.backbone.named_parameters() if "layer4" in n],
                 'lr': LEARNING_RATE * 0.05},
                {'params': [p for n, p in model.backbone.named_parameters() if "layer3" in n],
                 'lr': LEARNING_RATE * 0.02},
                {'params': [p for n, p in model.backbone.named_parameters() if "layer2" in n],
                 'lr': LEARNING_RATE * 0.01},
                {'params': [p for n, p in model.backbone.named_parameters() if "layer1" in n],
                 'lr': LEARNING_RATE * 0.005},
                {'params': [p for n, p in model.backbone.named_parameters()
                            if not any(x in n for x in ["layer1", "layer2", "layer3", "layer4"])],
                 'lr': LEARNING_RATE * 0.001},
                {'params': model.classifier.parameters(), 'lr': LEARNING_RATE * 0.3}
            ]
            optimizer = optim.AdamW(params, lr=LEARNING_RATE * 0.3, weight_decay=WEIGHT_DECAY * 0.6)

            # Aplicar clipping de gradiente mais forte
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            # Reiniciar scheduler
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[LEARNING_RATE * 0.05, LEARNING_RATE * 0.02, LEARNING_RATE * 0.01,
                        LEARNING_RATE * 0.005, LEARNING_RATE * 0.001, LEARNING_RATE * 0.3],
                total_steps=len(train_loader) * (epochs - epoch),
                pct_start=0.3,
                div_factor=10.0,
                final_div_factor=100.0
            )
            print(f"   - Otimizador e scheduler reiniciados com taxas de aprendizado mais conservadoras")

        elif epoch == unfreeze_stage4:
            print(f"\n🔓 Descongelando backbone inteiro (época {epoch + 1})...")
            # Descongelar todas as camadas
            for param in model.backbone.parameters():
                param.requires_grad = True

            # Reiniciar otimizador com taxas de aprendizado muito baixas para o backbone
            params = [
                {'params': [p for n, p in model.backbone.named_parameters() if "layer4" in n],
                 'lr': LEARNING_RATE * 0.03},  # Muito menor que antes
                {'params': [p for n, p in model.backbone.named_parameters() if "layer3" in n],
                 'lr': LEARNING_RATE * 0.01},
                {'params': [p for n, p in model.backbone.named_parameters() if "layer2" in n],
                 'lr': LEARNING_RATE * 0.005},
                {'params': [p for n, p in model.backbone.named_parameters() if "layer1" in n],
                 'lr': LEARNING_RATE * 0.002},
                {'params': [p for n, p in model.backbone.named_parameters()
                            if not any(x in n for x in ["layer1", "layer2", "layer3", "layer4"])],
                 'lr': LEARNING_RATE * 0.001},
                {'params': model.classifier.parameters(), 'lr': LEARNING_RATE * 0.1}
            ]
            optimizer = optim.AdamW(params, lr=LEARNING_RATE * 0.1, weight_decay=WEIGHT_DECAY * 0.4)

            # Aplicar clipping de gradiente mais forte
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[LEARNING_RATE * 0.03, LEARNING_RATE * 0.01, LEARNING_RATE * 0.005,
                        LEARNING_RATE * 0.002, LEARNING_RATE * 0.001, LEARNING_RATE * 0.1],
                total_steps=len(train_loader) * (epochs - epoch),
                pct_start=0.2,  # Aquecimento mais rápido
                div_factor=10.0,
                final_div_factor=100.0
            )
            print(f"   - Otimizador e scheduler reiniciados com taxas de aprendizado muito conservadoras")

        # Fase de treinamento
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Aplicar Mixup/CutMix se ativado
            if use_mixup_cutmix:
                inputs, labels_a, labels_b, lam = mixup_cutmix(inputs, labels)

                optimizer.zero_grad()
                outputs = model(inputs)

                # Calcular perda com rótulos misturados
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

                # Para cálculo de acurácia, usar o rótulo com maior peso
                if lam > 0.5:
                    target_for_acc = labels_a
                else:
                    target_for_acc = labels_b
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                target_for_acc = labels

            loss.backward()

            # Aplicar clipping de gradiente para prevenir explosão de gradiente
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()  # Atualizar a taxa de aprendizado por batch

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(target_for_acc).sum().item()

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

        # Atualizar histórico
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Salvar o melhor modelo baseado em acurácia
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"✅ Novo melhor modelo (acurácia) salvo! Acurácia de validação: {val_acc:.2f}%")
            counter = 0  # Reiniciar contador de early stopping
        else:
            counter += 1

        # Salvar também o melhor modelo baseado em perda (pode ser diferente)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_loss = model.state_dict().copy()
            print(f"✅ Novo melhor modelo (perda) salvo! Perda de validação: {val_loss:.4f}")

        # Imprimir progresso
        print(f'Época {epoch + 1}/{epochs} | '
              f'Treino: Perda {train_loss:.4f}, Acurácia {train_acc:.2f}% | '
              f'Validação: Perda {val_loss:.4f}, Acurácia {val_acc:.2f}%')

        # Early stopping
        if counter >= patience:
            print(f'⚠️ Early stopping ativado (sem melhoria por {patience} épocas)')
            break

        # Salvar checkpoint a cada 5 épocas
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
            }, checkpoint_path)
            print(f"💾 Checkpoint salvo: {checkpoint_path}")

    # Comparar qual modelo usar baseado em acurácia ou perda
    print("\n🔍 Comparando modelos baseados em acurácia e perda...")

    # Carregar modelo com melhor acurácia
    model_acc = model
    model_acc.load_state_dict(best_model_state)

    # Criar uma cópia do modelo com melhor perda
    model_loss = SnakeBinaryClassifier(base_model=model.base_model_name, freeze_backbone=False,
                                       dropout_rate=DROPOUT_RATE)
    model_loss.to(device)
    model_loss.load_state_dict(best_model_state_loss)

    # Avaliar na validação
    model_acc.eval()
    model_loss.eval()

    val_correct_acc = 0
    val_correct_loss = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Modelo com melhor acurácia
            outputs_acc = model_acc(inputs)
            _, predicted_acc = outputs_acc.max(1)

            # Modelo com melhor perda
            outputs_loss = model_loss(inputs)
            _, predicted_loss = outputs_loss.max(1)

            val_total += labels.size(0)
            val_correct_acc += predicted_acc.eq(labels).sum().item()
            val_correct_loss += predicted_loss.eq(labels).sum().item()

    final_acc_acc = 100. * val_correct_acc / val_total
    final_acc_loss = 100. * val_correct_loss / val_total

    print(f"📊 Modelo com melhor acurácia: {final_acc_acc:.2f}%")
    print(f"📊 Modelo com melhor perda: {final_acc_loss:.2f}%")

    # Escolher o modelo final baseado na comparação
    if final_acc_acc >= final_acc_loss:
        print("✅ Usando modelo com melhor acurácia como modelo final")
        final_model = model_acc
        final_state = best_model_state
    else:
        print("✅ Usando modelo com melhor perda como modelo final")
        final_model = model_loss
        final_state = best_model_state_loss

    # Gerar curvas de aprendizado
    generate_learning_curves("Classificação Binária", history, output_dir)

    return final_model, history, max(final_acc_acc, final_acc_loss)


def train_with_kfold(data_dir, output_dir, model_name, epochs, batch_size, label_smoothing, k=5):
    """
    Treina o modelo usando validação cruzada k-fold.

    A validação cruzada k-fold divide os dados em k partes, treinando
    k modelos diferentes, cada um usando k-1 partes para treino e 1 para validação.
    Isso proporciona uma avaliação mais robusta do desempenho do modelo.

    Args:
        data_dir: Diretório contendo as imagens
        output_dir: Diretório para salvar resultados
        model_name: Nome do modelo base a ser usado
        epochs: Número de épocas por fold
        batch_size: Tamanho do batch
        label_smoothing: Valor para label smoothing
        k: Número de folds

    Returns:
        Dicionário com resultados médios e por fold
    """
    print(f"\n{'=' * 80}")
    print(f"🔄 INICIANDO TREINAMENTO COM {k}-FOLD CROSS VALIDATION")
    print(f"{'=' * 80}")

    # Criar diretório para folds
    folds_dir = os.path.join(output_dir, FOLDS_DIR)
    os.makedirs(folds_dir, exist_ok=True)

    # Obter todos os caminhos de imagens e rótulos
    all_image_paths = []
    all_labels = []

    # Mapeamento de classes
    class_mapping = {
        "naopeconhentas": 0,
        "peconhentas": 1
    }

    # Coleta dados de todas as imagens
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Verificar se a classe é válida
        if class_name.lower() not in class_mapping:
            print(f"⚠️ Classe não reconhecida: {class_name}. Pulando.")
            continue

        class_idx = class_mapping[class_name.lower()]

        # Coleta paths de imagens
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                all_image_paths.append(img_path)
                all_labels.append(class_idx)

    # Converter para numpy arrays para uso com KFold
    all_image_paths = np.array(all_image_paths)
    all_labels = np.array(all_labels)

    # Tabular classes e quantidade
    class_counts = Counter(all_labels)
    print("\nDistribuição de classes no dataset completo:")
    for class_idx, count in class_counts.items():
        class_name = list(class_mapping.keys())[list(class_mapping.values()).index(class_idx)]
        print(f"  - {class_name} (índice {class_idx}): {count} imagens")

    # Inicializar KFold com stratify para manter distribuição de classes
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    # Resultados por fold
    fold_results = []
    fold_models = []

    # Para cada fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_image_paths)):
        print(f"\n{'=' * 80}")
        print(f"🔄 TREINANDO FOLD {fold + 1}/{k}")
        print(f"{'=' * 80}")

        # Separar dados para este fold
        train_paths = all_image_paths[train_idx].tolist()
        train_labels = all_labels[train_idx].tolist()
        val_paths = all_image_paths[val_idx].tolist()
        val_labels = all_labels[val_idx].tolist()

        # Verificar distribuição de classes neste fold
        train_class_dist = Counter(train_labels)
        val_class_dist = Counter(val_labels)

        print(f"\nDistribuição de classes para Fold {fold + 1}:")
        print("  - Conjunto de Treino:")
        for class_idx, count in train_class_dist.items():
            class_name = list(class_mapping.keys())[list(class_mapping.values()).index(class_idx)]
            print(f"      {class_name} (índice {class_idx}): {count} imagens")

        print("  - Conjunto de Validação:")
        for class_idx, count in val_class_dist.items():
            class_name = list(class_mapping.keys())[list(class_mapping.values()).index(class_idx)]
            print(f"      {class_name} (índice {class_idx}): {count} imagens")

        # Criar diretório para este fold
        fold_dir = os.path.join(folds_dir, f"fold_{fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)

        # Obter transformações
        train_transform, val_transform, _ = get_advanced_transforms()

        # Criar datasets
        train_dataset = SnakeDataset(train_paths, train_labels, train_transform)
        val_dataset = SnakeDataset(val_paths, val_labels, val_transform)

        # Calcular pesos para balanceamento
        class_weights = torch.FloatTensor([1.0, VENOMOUS_WEIGHT]).to(DEVICE)

        # Criar data loaders
        if VENOMOUS_WEIGHT > 1.0:
            # Usar WeightedRandomSampler para classes desbalanceadas
            sample_weights = [VENOMOUS_WEIGHT if label == 1 else 1.0 for label in train_labels]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_labels), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                      num_workers=NUM_WORKERS, pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=NUM_WORKERS, pin_memory=True)

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

        # Criar modelo
        model = SnakeBinaryClassifier(base_model=model_name, freeze_backbone=True, dropout_rate=DROPOUT_RATE)
        model = model.to(DEVICE)

        # Otimizador
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

        # Critério com label smoothing
        if label_smoothing > 0:
            print(f"🔄 Usando Label Smoothing com valor {label_smoothing}")
            criterion = LabelSmoothingLoss(classes=2, smoothing=label_smoothing, weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Treinar modelo para este fold
        fold_model, fold_history, fold_val_acc = train_with_progressive_unfreezing(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=DEVICE,
            epochs=epochs,
            class_names=list(class_mapping.keys()),
            output_dir=fold_dir,
            use_mixup_cutmix=USE_MIXUP_CUTMIX
        )

        # Salvar modelo deste fold
        fold_model_path = os.path.join(fold_dir, f"model_fold_{fold + 1}.pth")
        torch.save({
            'model_state_dict': fold_model.state_dict(),
            'history': fold_history,
            'val_acc': fold_val_acc,
            'fold': fold + 1,
            'class_names': list(class_mapping.keys()),
            'class_to_idx': class_mapping,
            'model_config': {
                'base_model': model_name,
                'binary': True,
                'dropout_rate': DROPOUT_RATE,
                'use_mixup_cutmix': USE_MIXUP_CUTMIX
            }
        }, fold_model_path)

        # Registrar resultados
        fold_results.append({
            'fold': fold + 1,
            'val_acc': fold_val_acc,
            'history': fold_history,
            'model_path': fold_model_path
        })

        # Salvar modelo para possível ensemble
        fold_models.append(fold_model)

        print(f"✅ Fold {fold + 1} concluído. Acurácia de validação: {fold_val_acc:.2f}%")

    # Calcular métricas médias
    avg_val_acc = sum(result['val_acc'] for result in fold_results) / k

    print(f"\n{'=' * 80}")
    print(f"📊 RESULTADOS DA VALIDAÇÃO CRUZADA {k}-FOLD")
    print(f"{'=' * 80}")
    print(f"Acurácia média: {avg_val_acc:.2f}%")

    for fold_result in fold_results:
        print(f"Fold {fold_result['fold']}: {fold_result['val_acc']:.2f}%")

    # Criar e salvar gráfico de comparação
    plt.figure(figsize=(12, 6))

    for fold_result in fold_results:
        plt.plot(fold_result['history']['val_acc'],
                 label=f"Fold {fold_result['fold']} ({fold_result['val_acc']:.1f}%)")

    plt.axhline(y=avg_val_acc, color='r', linestyle='--',
                label=f'Média ({avg_val_acc:.2f}%)')

    plt.title(f'Acurácia de Validação por Fold ({k}-Fold Cross Validation)')
    plt.xlabel('Época')
    plt.ylabel('Acurácia (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'kfold_validation_accuracy.png'))

    # Salvar resultados
    kfold_results = {
        'avg_val_acc': float(avg_val_acc),
        'fold_results': [
            {
                'fold': res['fold'],
                'val_acc': float(res['val_acc']),
                'model_path': res['model_path']
            } for res in fold_results
        ],
        'class_mapping': class_mapping
    }

    # Salvar resultados como JSON
    with open(os.path.join(output_dir, 'kfold_results.json'), 'w') as f:
        json.dump(kfold_results, f, indent=4)

    # Selecionar melhor modelo como modelo final (ou usar ensemble posteriormente)
    best_fold_idx = np.argmax([res['val_acc'] for res in fold_results])
    best_fold = fold_results[best_fold_idx]

    print(f"\n✅ Melhor modelo: Fold {best_fold['fold']} com acurácia {best_fold['val_acc']:.2f}%")

    # Copiar o melhor modelo para o diretório principal
    best_model_path = os.path.join(output_dir, f"best_fold_{best_fold['fold']}_model.pth")
    shutil.copy2(best_fold['model_path'], best_model_path)

    return {
        'avg_val_acc': avg_val_acc,
        'fold_results': fold_results,
        'best_fold': best_fold['fold'],
        'best_model_path': best_model_path,
        'fold_models': fold_models
    }


def evaluate_model_with_thresholds(model, test_loader, criterion, device, class_names, output_dir):
    """
    Avalia o modelo no conjunto de teste e otimiza thresholds.

    Realiza uma avaliação completa do modelo no conjunto de teste,
    incluindo otimização de thresholds para melhorar a detecção de
    serpentes peçonhentas, que é crítico para segurança.

    Args:
        model: Modelo treinado para avaliação
        test_loader: DataLoader com dados de teste
        criterion: Função de perda
        device: Dispositivo (CPU/GPU)
        class_names: Nomes das classes
        output_dir: Diretório para salvar resultados

    Returns:
        Acurácia no teste, relatório de classificação e thresholds ótimos
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    y_true = []
    y_pred = []
    y_scores = []  # Probabilidades para a classe positiva (peçonhenta)

    # Registrar previsões por classe - para análise de erros
    class_predictions = {
        "naopeconhentas": {"correct": 0, "total": 0, "confidences": []},
        "peconhentas": {"correct": 0, "total": 0, "confidences": []}
    }

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            # Obter probabilidades
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            venomous_prob = probabilities[:, 1]  # Probabilidade da classe peçonhenta

            # Previsão baseada no threshold padrão (0.5)
            _, predicted = outputs.max(1)

            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            # Registrar resultados para análise de erros
            for i, (label, pred, prob) in enumerate(zip(labels, predicted, venomous_prob)):
                class_idx = label.item()
                class_name = class_names[class_idx]

                class_predictions[class_name]["total"] += 1
                if pred.item() == class_idx:
                    class_predictions[class_name]["correct"] += 1

                # Armazenar a confiança correta (probabilidade da classe verdadeira)
                if class_idx == 1:  # Peçonhenta
                    class_predictions[class_name]["confidences"].append(prob.item())
                else:  # Não peçonhenta
                    class_predictions[class_name]["confidences"].append(1 - prob.item())

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(venomous_prob.cpu().numpy())

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * test_correct / test_total

    # Gerar relatório de classificação com threshold padrão (0.5)
    report = classification_report(y_true, y_pred,
                                   target_names=class_names,
                                   output_dict=True)

    # Criar matriz de confusão com threshold padrão
    cm = confusion_matrix(y_true, y_pred)

    # Criar e salvar visualização da matriz de confusão
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Matriz de Confusão - Classificação Binária (Threshold: 0.5)')
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Previsto')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_default.png'))
    plt.close()

    # Calcular AUC-ROC e curva de Precision-Recall
    roc_auc, avg_precision = plot_roc_curve(model, test_loader, device, class_names, output_dir)

    # Encontrar thresholds ótimos baseados em diferentes critérios
    optimal_thresholds = find_optimal_threshold(y_true, y_scores, output_dir)

    # Avaliar com threshold otimizado para alta recall (prioridade para não perder cobras peçonhentas)
    high_recall_threshold = optimal_thresholds['high_recall']
    y_pred_optimized = [1 if score >= high_recall_threshold else 0 for score in y_scores]

    # Criar matriz de confusão com threshold otimizado
    cm_optimized = confusion_matrix(y_true, y_pred_optimized)
    report_optimized = classification_report(y_true, y_pred_optimized,
                                             target_names=class_names,
                                             output_dict=True)

    # Visualizar matriz de confusão com threshold otimizado
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Matriz de Confusão - Threshold Otimizado ({high_recall_threshold:.3f})')
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Previsto')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_optimized.png'))
    plt.close()

    # Gráfico de distribuição de confiança por classe
    plt.figure(figsize=(12, 6))

    # Confidências para não peçonhentas
    confidences_non_venomous = class_predictions["naopeconhentas"]["confidences"]

    # Confidências para peçonhentas
    confidences_venomous = class_predictions["peconhentas"]["confidences"]

    plt.hist(confidences_non_venomous, alpha=0.5, bins=20,
             label=f'Não Peçonhentas (n={len(confidences_non_venomous)})',
             color='green')
    plt.hist(confidences_venomous, alpha=0.5, bins=20,
             label=f'Peçonhentas (n={len(confidences_venomous)})',
             color='red')

    # Adicionar linhas verticais para thresholds
    plt.axvline(x=0.5, color='black', linestyle='--',
                label='Threshold Padrão (0.5)')
    plt.axvline(x=high_recall_threshold, color='purple', linestyle='--',
                label=f'Threshold Alta Recall ({high_recall_threshold:.3f})')

    plt.xlabel('Confiança na Classe Correta')
    plt.ylabel('Número de Imagens')
    plt.title('Distribuição de Confiança por Classe')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
    plt.close()

    # Calcular acurácia com threshold otimizado
    acc_optimized = np.mean(np.array(y_true) == np.array(y_pred_optimized)) * 100

    print(f"\n{'=' * 80}")
    print(f"📊 RESULTADOS DA AVALIAÇÃO NO CONJUNTO DE TESTE")
    print(f"{'=' * 80}")
    print(f"Acurácia (threshold 0.5): {test_acc:.2f}%")
    print(f"Acurácia (threshold {high_recall_threshold:.3f}): {acc_optimized:.2f}%")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")

    print(f"\nMatriz de Confusão (threshold 0.5):")
    print(cm)

    print(f"\nMatriz de Confusão (threshold otimizado {high_recall_threshold:.3f}):")
    print(cm_optimized)

    print("\nThresholds ótimos encontrados:")
    for criterion, threshold in optimal_thresholds.items():
        print(f"  - {criterion}: {threshold:.3f}")

    print("\nDesempenho por classe (threshold 0.5):")
    for i, cls in enumerate(class_names):
        print(f"\nMétricas para '{cls}':")
        print(f"  Precisão: {report[cls]['precision']:.4f}")
        print(f"  Recall: {report[cls]['recall']:.4f}")
        print(f"  F1-Score: {report[cls]['f1-score']:.4f}")
        correct = class_predictions[cls]["correct"]
        total = class_predictions[cls]["total"]
        print(f"  Acurácia: {correct}/{total} ({100.0 * correct / total:.2f}%)")

    print("\nDesempenho por classe (threshold otimizado):")
    for i, cls in enumerate(class_names):
        print(f"\nMétricas para '{cls}':")
        print(f"  Precisão: {report_optimized[cls]['precision']:.4f}")
        print(f"  Recall: {report_optimized[cls]['recall']:.4f}")
        print(f"  F1-Score: {report_optimized[cls]['f1-score']:.4f}")

    # Salvar relatório completo
    with open(os.path.join(output_dir, 'test_report.json'), 'w') as f:
        json.dump({
            'accuracy_default': test_acc,
            'accuracy_optimized': acc_optimized,
            'auc_roc': roc_auc,
            'average_precision': avg_precision,
            'optimal_thresholds': optimal_thresholds,
            'confusion_matrix_default': cm.tolist(),
            'confusion_matrix_optimized': cm_optimized.tolist(),
            'classification_report_default': report,
            'classification_report_optimized': report_optimized
        }, f, indent=4)

    return test_acc, report, optimal_thresholds


def main():
    """
    Função principal para treinamento do modelo.

    Coordena todo o processo de treinamento, desde o processamento dos argumentos
    de linha de comando, preparação dos dados, treinamento do modelo até a avaliação
    final e salvamento dos resultados.
    """
    # Configuração do parser de argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Treinar modelo de classificação binária de serpentes')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Diretório com as imagens (pastas peconhentas/naopeconhentas)')
    parser.add_argument('--output-dir', type=str, default='resultados_snake',
                        help='Diretório para salvar resultados')
    parser.add_argument('--augmented-dir', type=str, default=None,
                        help='Diretório para dados aumentados')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'vgg16', 'efficientnet_b3', 'densenet169'],
                        help='Modelo base a ser usado')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Número de épocas para treinar')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Tamanho do batch para treinamento')
    parser.add_argument('--skip-augmentation', action='store_true',
                        help='Pular a etapa de aumentação de dados')
    parser.add_argument('--kfold', type=int, default=0,
                        help='Número de folds para validação cruzada (0 = desativado)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Valor para label smoothing (0 = desativado)')
    parser.add_argument('--use-mixup-cutmix', action='store_true',
                        help='Ativar técnicas de Mixup e CutMix')
    parser.add_argument('--no-mixup-cutmix', action='store_true',
                        help='Desativar técnicas de Mixup e CutMix')

    # Processar argumentos da linha de comando
    args = parser.parse_args()

    # Definir o valor de use_mixup_cutmix localmente
    use_mixup_cutmix = USE_MIXUP_CUTMIX  # Usar o valor global como padrão

    # Sobrescrever baseado nos argumentos
    if args.use_mixup_cutmix:
        use_mixup_cutmix = True
    if args.no_mixup_cutmix:
        use_mixup_cutmix = False

    # Adicionar timestamp ao diretório de saída para evitar sobrescrita
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"snake_{args.model}_{timestamp}")

    # Exibir cabeçalho e informações de configuração
    print("\n" + "=" * 80)
    print(f"🐍 TREINAMENTO DE MODELO PARA CLASSIFICAÇÃO DE SERPENTES 🐍".center(80))
    print("=" * 80 + "\n")

    print(f"📋 Configurações:")
    print(f"  - Modelo base: {args.model}")
    print(f"  - Épocas: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Dropout: {DROPOUT_RATE}")
    print(f"  - Peso extra para peçonhentas: {VENOMOUS_WEIGHT}")
    print(f"  - Aumento extra para peçonhentas: {EXTRA_VENOMOUS_AUGMENTATION}")
    print(f"  - Augmentation forte: {STRONG_AUGMENTATION}")
    print(f"  - Label Smoothing: {args.label_smoothing}")
    print(f"  - Mixup e CutMix: {use_mixup_cutmix}")

    if args.kfold > 1:
        print(f"  - Validação Cruzada: {args.kfold}-fold")
    else:
        print(f"  - Validação Cruzada: Desativada")

    # Verificar se devemos usar dados aumentados existentes ou gerar novos
    data_dir = args.data_dir

    # Realizar aumentação de dados se não for pulada
    if not args.skip_augmentation:
        # Definir diretório para dados aumentados
        if args.augmented_dir is None:
            augmented_dir = f"augmented_snake_{timestamp}"
        else:
            augmented_dir = args.augmented_dir

        print(f"\n🔄 Gerando dados aumentados em: {augmented_dir}")
        augment_and_save_images(data_dir, augmented_dir, TARGET_IMAGES_PER_CLASS)
        data_dir = augmented_dir  # Atualizar diretório de dados para usar os aumentados

    # Criar diretórios de saída
    os.makedirs(output_dir, exist_ok=True)

    # Se estamos usando k-fold, precisamos do diretório para folds
    create_output_directories(output_dir, with_folds=(args.kfold > 1))

    # Escolher entre treinamento normal ou k-fold
    if args.kfold > 1:
        # Treinamento com validação cruzada k-fold
        kfold_results = train_with_kfold(
            data_dir=data_dir,
            output_dir=output_dir,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            label_smoothing=args.label_smoothing,
            k=args.kfold
        )

        print(f"\n✅ Treinamento com validação cruzada {args.kfold}-fold concluído!")
        print(f"📊 Acurácia média: {kfold_results['avg_val_acc']:.2f}%")
        print(
            f"📊 Melhor fold: {kfold_results['best_fold']} - Acurácia: {kfold_results['fold_results'][kfold_results['best_fold'] - 1]['val_acc']:.2f}%")

        # Criar versão simplificada do melhor modelo
        best_model_path = kfold_results['best_model_path']
        simple_model_path = os.path.join(args.output_dir, f"snake_model_{args.model}_kfold.pth")
        shutil.copy2(best_model_path, simple_model_path)

        print(f"\n✅ Melhor modelo salvo em: {best_model_path}")
        print(f"✅ Modelo simplificado salvo em: {simple_model_path}")

    else:
        # Treinamento normal (single fold)
        # Preparar dados
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names, class_to_idx = prepare_data_with_test_set(
            data_dir, min_images_per_class=20
        )

        # Definir transformações para cada conjunto de dados
        train_transform, val_transform, test_transform = get_advanced_transforms()

        # Criar datasets para treino, validação e teste
        train_dataset = SnakeDataset(train_paths, train_labels, train_transform)
        val_dataset = SnakeDataset(val_paths, val_labels, val_transform)
        test_dataset = SnakeDataset(test_paths, test_labels, test_transform)

        # Calcular pesos de classe para lidar com possível desbalanceamento
        weights = [1.0, VENOMOUS_WEIGHT]  # Peso fixo maior para a classe peçonhenta
        class_weights = torch.FloatTensor(weights).to(DEVICE)

        print("\n📊 Pesos de classe para balanceamento:")
        for i, cls in enumerate(class_names):
            print(f"  - {cls}: {class_weights[i]:.4f}")

        # Criar samplers para balanceamento
        if VENOMOUS_WEIGHT > 1.0:
            print("\n🔄 Usando WeightedRandomSampler para equilibrar classes durante o treinamento")
            sample_weights = [weights[label] for label in train_labels]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_labels), replacement=True)
            shuffle = False  # Não usar shuffle quando usando sampler
        else:
            sampler = None
            shuffle = True

        # Criar data loaders para treino, validação e teste
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        # Inicializar modelo com backbone pré-treinado
        model = SnakeBinaryClassifier(base_model=args.model, freeze_backbone=True, dropout_rate=DROPOUT_RATE)
        model = model.to(DEVICE)  # Transferir modelo para GPU se disponível

        # Configurar otimizador
        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        # Configurar critério (função de perda) com label smoothing
        if args.label_smoothing > 0:
            print(f"\n🔄 Usando Label Smoothing com valor {args.label_smoothing}")
            criterion = LabelSmoothingLoss(classes=2, smoothing=args.label_smoothing, weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Treinar com descongelamento progressivo do backbone
        trained_model, history, best_val_acc = train_with_progressive_unfreezing(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=DEVICE,
            epochs=args.epochs,
            class_names=class_names,
            output_dir=output_dir,
            use_mixup_cutmix=use_mixup_cutmix  # Passando a variável local
        )

        # Avaliar no conjunto de teste com análise de thresholds
        test_acc, test_report, optimal_thresholds = evaluate_model_with_thresholds(
            model=trained_model,
            test_loader=test_loader,
            criterion=criterion,
            device=DEVICE,
            class_names=class_names,
            output_dir=output_dir
        )

        # Salvar o modelo treinado com todas as informações
        model_save_path = os.path.join(output_dir, "models", f"snake_binary_{args.model}.pth")
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'class_names': class_names,
            'class_to_idx': class_to_idx,
            'history': history,
            'model_config': {
                'base_model': args.model,
                'binary': True,
                'dropout_rate': DROPOUT_RATE,
                'optimal_thresholds': optimal_thresholds,
                'label_smoothing': args.label_smoothing,
                'use_mixup_cutmix': use_mixup_cutmix
            }
        }, model_save_path)

        # Criar uma versão simplificada do modelo para uso mais fácil
        simple_model_path = os.path.join(args.output_dir, f"snake_model_{args.model}.pth")
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'class_names': class_names,
            'class_to_idx': class_to_idx,
            'model_config': {
                'base_model': args.model,
                'binary': True,
                'optimal_thresholds': optimal_thresholds
            }
        }, simple_model_path)

        # Exibir resumo dos resultados
        print(f"\n✅ Modelo completo salvo em: {model_save_path}")
        print(f"✅ Modelo simplificado salvo em: {simple_model_path}")
        print("\n📊 RESUMO FINAL:")
        print(f"  - Modelo base: {args.model}")
        print(f"  - Melhor acurácia (validação): {best_val_acc:.2f}%")
        print(f"  - Acurácia no teste (threshold padrão): {test_acc:.2f}%")
        print(f"  - Threshold recomendado para alta segurança: {optimal_thresholds['high_recall']:.3f}")
        print(f"  - Análise detalhada em: {output_dir}")

    # Exemplo de como usar o modelo treinado para inferência
    print("\n💡 Para inferência, use o script de inferência com:")
    print(f"  python snake_inference.py --model {simple_model_path} --high-recall --dir sua_pasta_com_imagens")

if __name__ == "__main__":
    main()