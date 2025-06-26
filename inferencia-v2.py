import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import sys
import argparse
from pathlib import Path
import matplotlib

# Set matplotlib backend to non-interactive mode to prevent windows from opening
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision import models
import time


class SnakeCNN_TransferLearning(nn.Module):
    """
    Modelo para classificação de serpentes usando transfer learning
    com diferentes backbones pré-treinados.
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

        # Estrutura do classificador depende se estamos usando o modelo melhorado ou original
        if num_classes <= 30:  # Assumindo que é o modelo original se num_classes for razoável
            try:
                # Tentar carregar com a estrutura melhorada (mais camadas)
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(feature_size, 512),  # Aumentado para 512 neurônios
                    nn.BatchNorm1d(512),  # BatchNorm
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),  # Dropout aumentado
                    nn.Linear(512, 256),  # Camada intermediária
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
            except:
                # Fallback para a estrutura original
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(feature_size, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(128, num_classes)
                )
        else:
            # Para compatibilidade com modelos antigos
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feature_size, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )

    def forward(self, x):
        # Extração de features pelo backbone pré-treinado
        x = self.backbone(x)
        # Adaptação do tamanho
        x = self.adaptive_pool(x)
        # Classificação
        x = self.classifier(x)
        return x


class UNetEncoder(nn.Module):
    """Encoder do U-Net adaptado para classificação"""

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
    """

    def __init__(self, num_classes):
        super(SnakeCNN_UNet, self).__init__()

        # Usar o encoder do U-Net
        self.backbone = UNetEncoder()

        # Adaptação para extrair features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Estrutura do classificador depende se estamos usando o modelo melhorado ou original
        if num_classes <= 30:  # Assumindo que é o modelo original se num_classes for razoável
            try:
                # Tentar carregar com a estrutura melhorada (mais camadas)
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(1024, 512),  # U-Net bottleneck tem 1024 canais
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
            except:
                # Fallback para a estrutura original
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(1024, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(128, num_classes)
                )
        else:
            # Para compatibilidade com modelos antigos
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1024, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )

    def forward(self, x):
        x = self.backbone(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


def get_model(model_type, num_classes):
    """
    Retorna um modelo baseado no tipo especificado.

    Args:
        model_type: Tipo do modelo ('resnet50', 'vgg16', etc.)
        num_classes: Número de classes

    Returns:
        model: Modelo inicializado
    """
    if model_type == 'unet':
        return SnakeCNN_UNet(num_classes)
    else:
        return SnakeCNN_TransferLearning(num_classes, base_model=model_type)


def load_model(model_path):
    """
    Carrega um modelo treinado.

    Args:
        model_path: Caminho para o arquivo do modelo (.pth)

    Returns:
        model: Modelo carregado
        class_names: Nomes das classes
        transform: Transformações para pré-processamento de imagens
    """
    print(f"📂 Carregando modelo de {model_path}")

    # Verifica se o arquivo existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {model_path}")

    # Carrega o checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)

    # Extrai informações sobre o modelo
    model_config = checkpoint.get('model_config', {})
    class_names = checkpoint.get('class_names', [])

    if not class_names:
        raise ValueError("Não foi possível extrair os nomes das classes do modelo.")

    # Determina o tipo do modelo
    base_model = model_config.get('base_model', None)

    if base_model is None:
        # Tenta extrair do nome do arquivo
        file_name = os.path.basename(model_path)
        for model_type in ['resnet50', 'vgg16', 'efficientnet_b3', 'densenet169', 'inception_v3', 'unet']:
            if model_type in file_name.lower():
                base_model = model_type
                break

    if base_model is None:
        raise ValueError("Não foi possível determinar o tipo do modelo.")

    # Inicializa o modelo
    num_classes = len(class_names)
    print(f"🧠 Inicializando modelo {base_model} para {num_classes} classes")

    model = get_model(base_model, num_classes)

    # Carrega os pesos
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Define o modelo para modo de avaliação

    # Define a transformação adequada
    if base_model == 'inception_v3':
        image_size = 299  # Inception v3 usa 299x299
    else:
        image_size = 224  # Outros modelos usam 224x224

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return model, class_names, transform, base_model


def load_ensemble(ensemble_path):
    """
    Carrega um ensemble de modelos.

    Args:
        ensemble_path: Caminho para o arquivo de configuração do ensemble

    Returns:
        models: Lista de modelos carregados
        class_names: Nomes das classes
        transforms: Lista de transformações para cada modelo
    """
    print(f"📂 Carregando ensemble de {ensemble_path}")

    # Verifica se o arquivo existe
    if not os.path.exists(ensemble_path):
        raise FileNotFoundError(f"Arquivo de ensemble não encontrado: {ensemble_path}")

    # Carrega a configuração do ensemble
    with open(ensemble_path, 'r') as f:
        ensemble_config = json.load(f)

    model_paths = ensemble_config.get('model_paths', [])
    class_names = ensemble_config.get('class_names', [])

    if not model_paths:
        raise ValueError("Não foram encontrados caminhos de modelos na configuração do ensemble.")

    if not class_names:
        raise ValueError("Não foram encontrados nomes de classes na configuração do ensemble.")

    # Carrega cada modelo do ensemble
    models = []
    transforms_list = []
    model_types = []

    for model_path in model_paths:
        model, _, transform, model_type = load_model(model_path)
        models.append(model)
        transforms_list.append(transform)
        model_types.append(model_type)

    print(f"✅ Ensemble carregado com {len(models)} modelos: {', '.join(model_types)}")

    return models, class_names, transforms_list


def load_image(image_path, transform):
    """
    Carrega e pré-processa uma imagem.

    Args:
        image_path: Caminho para a imagem
        transform: Transformações a serem aplicadas

    Returns:
        original_image: Imagem original para visualização
        input_tensor: Tensor pré-processado para o modelo
    """
    try:
        # Carrega a imagem
        original_image = Image.open(image_path).convert('RGB')

        # Aplica as transformações
        input_tensor = transform(original_image)

        # Adiciona dimensão de batch
        input_tensor = input_tensor.unsqueeze(0)

        return original_image, input_tensor

    except Exception as e:
        print(f"❌ Erro ao carregar a imagem {image_path}: {e}")
        return None, None


def predict_with_model(model, input_tensor, class_names, topk=3):
    """
    Faz uma predição usando um modelo.

    Args:
        model: Modelo para fazer a predição
        input_tensor: Tensor de entrada
        class_names: Nomes das classes
        topk: Número de top-k predições a retornar

    Returns:
        topk_probs: Probabilidades das top-k predições
        topk_classes: Nomes das classes correspondentes
    """
    with torch.no_grad():
        model.eval()
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    # Obtém as top-k probabilidades e índices
    topk_probs, topk_indices = torch.topk(probabilities, k=min(topk, len(class_names)))

    # Converte para numpy
    topk_probs = topk_probs.cpu().numpy()
    topk_indices = topk_indices.cpu().numpy()

    # Obtém os nomes das classes
    topk_classes = [class_names[idx] for idx in topk_indices]

    return topk_probs, topk_classes


def predict_with_ensemble(models, input_tensors, class_names, topk=3):
    """
    Faz uma predição usando um ensemble de modelos.

    Args:
        models: Lista de modelos para fazer a predição
        input_tensors: Lista de tensores de entrada (um para cada modelo)
        class_names: Nomes das classes
        topk: Número de top-k predições a retornar

    Returns:
        topk_probs: Probabilidades das top-k predições
        topk_classes: Nomes das classes correspondentes
    """
    all_probabilities = []

    # Obtém as probabilidades de cada modelo
    for model, input_tensor in zip(models, input_tensors):
        with torch.no_grad():
            model.eval()
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            all_probabilities.append(probabilities)

    # Calcula a média das probabilidades
    ensemble_probabilities = torch.stack(all_probabilities).mean(dim=0)

    # Obtém as top-k probabilidades e índices
    topk_probs, topk_indices = torch.topk(ensemble_probabilities, k=min(topk, len(class_names)))

    # Converte para numpy
    topk_probs = topk_probs.cpu().numpy()
    topk_indices = topk_indices.cpu().numpy()

    # Obtém os nomes das classes
    topk_classes = [class_names[idx] for idx in topk_indices]

    return topk_probs, topk_classes


def visualize_prediction(image, topk_probs, topk_classes, model_name, save_path=None):
    """
    Salva a visualização em arquivo se for requisitado, mas não mostra na tela.

    Args:
        image: Imagem original
        topk_probs: Probabilidades das top-k predições
        topk_classes: Nomes das classes correspondentes
        model_name: Nome do modelo
        save_path: Caminho para salvar a visualização

    Returns:
        None
    """
    # Apenas salvar se for especificado um caminho
    if save_path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Mostra a imagem
        ax1.imshow(image)
        ax1.set_title('Imagem Original')
        ax1.axis('off')

        # Mostra as predições
        colors = ['green' if p > 0.8 else 'orange' if p > 0.5 else 'red' for p in topk_probs]
        y_pos = range(len(topk_classes))

        bars = ax2.barh(y_pos, topk_probs * 100, align='center', color=colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(topk_classes)
        ax2.set_xlabel('Probabilidade (%)')
        ax2.set_title(f'Predições do Modelo: {model_name}')

        # Adiciona os valores de probabilidade nas barras
        for i, bar in enumerate(bars):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f'{topk_probs[i] * 100:.1f}%', va='center')

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"✅ Visualização salva em {save_path}")
        plt.close(fig)


def process_image(image_path, model, class_names, transform, model_name, topk=3, output_dir=None):
    """
    Processa uma imagem usando um modelo.

    Args:
        image_path: Caminho para a imagem
        model: Modelo para fazer a predição
        class_names: Nomes das classes
        transform: Transformações a serem aplicadas
        model_name: Nome do modelo
        topk: Número de top-k predições a retornar
        output_dir: Diretório para salvar a visualização

    Returns:
        topk_probs: Probabilidades das top-k predições
        topk_classes: Nomes das classes correspondentes
    """
    # Carrega a imagem
    original_image, input_tensor = load_image(image_path, transform)

    if original_image is None:
        return None, None

    # Faz a predição
    start_time = time.time()
    topk_probs, topk_classes = predict_with_model(model, input_tensor, class_names, topk)
    inference_time = (time.time() - start_time) * 1000  # ms

    # Imprime os resultados
    print(f"\nResultados para {os.path.basename(image_path)}:")
    for i, (cls, prob) in enumerate(zip(topk_classes, topk_probs)):
        print(f"{i + 1}. {cls}: {prob * 100:.2f}%")
    print(f"Tempo de inferência: {inference_time:.2f} ms")

    # Salva a visualização sem mostrar na tela
    if output_dir:
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(output_dir, f"{img_name}_{model_name}_prediction.png")
        visualize_prediction(original_image, topk_probs, topk_classes, model_name, save_path)

    return topk_probs, topk_classes


def process_image_with_ensemble(image_path, models, class_names, transforms_list, topk=3, output_dir=None):
    """
    Processa uma imagem usando um ensemble de modelos.

    Args:
        image_path: Caminho para a imagem
        models: Lista de modelos para fazer a predição
        class_names: Nomes das classes
        transforms_list: Lista de transformações a serem aplicadas (uma para cada modelo)
        topk: Número de top-k predições a retornar
        output_dir: Diretório para salvar a visualização

    Returns:
        topk_probs: Probabilidades das top-k predições
        topk_classes: Nomes das classes correspondentes
    """
    # Carrega a imagem para cada modelo com a transformação correspondente
    input_tensors = []
    original_image = None

    for transform in transforms_list:
        img, input_tensor = load_image(image_path, transform)

        if img is None:
            return None, None

        if original_image is None:
            original_image = img

        input_tensors.append(input_tensor)

    # Faz a predição com o ensemble
    start_time = time.time()
    topk_probs, topk_classes = predict_with_ensemble(models, input_tensors, class_names, topk)
    inference_time = (time.time() - start_time) * 1000  # ms

    # Imprime os resultados
    print(f"\nResultados do Ensemble para {os.path.basename(image_path)}:")
    for i, (cls, prob) in enumerate(zip(topk_classes, topk_probs)):
        print(f"{i + 1}. {cls}: {prob * 100:.2f}%")
    print(f"Tempo de inferência: {inference_time:.2f} ms")

    # Salva a visualização sem mostrar na tela
    if output_dir:
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(output_dir, f"{img_name}_ensemble_prediction.png")
        visualize_prediction(original_image, topk_probs, topk_classes, "Ensemble", save_path)

    return topk_probs, topk_classes


def process_directory(images_dir, model, class_names, transform, model_name, topk=3, output_dir=None):
    """
    Processa todas as imagens em um diretório.

    Args:
        images_dir: Diretório contendo as imagens
        model: Modelo para fazer a predição
        class_names: Nomes das classes
        transform: Transformações a serem aplicadas
        model_name: Nome do modelo
        topk: Número de top-k predições a retornar
        output_dir: Diretório para salvar as visualizações

    Returns:
        results: Resultados das predições
    """
    # Verifica se o diretório existe
    if not os.path.exists(images_dir):
        print(f"❌ Diretório não encontrado: {images_dir}")
        return None

    # Obtém a lista de imagens
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_files = [f for f in os.listdir(images_dir)
                   if os.path.isfile(os.path.join(images_dir, f)) and
                   f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"❌ Nenhuma imagem encontrada em {images_dir}")
        return None

    print(f"📁 Processando {len(image_files)} imagens de {images_dir}")

    # Cria o diretório de saída se necessário
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Resultados serão salvos em {output_dir}")

        # Cria um arquivo CSV para os resultados
        csv_path = os.path.join(output_dir, f"{model_name}_predictions.csv")
        with open(csv_path, 'w') as f:
            f.write("image,top1_class,top1_prob,top2_class,top2_prob,top3_class,top3_prob\n")

    # Processa cada imagem
    results = {}

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        print(f"[{i + 1}/{len(image_files)}] Processando {image_file}...")

        topk_probs, topk_classes = process_image(
            image_path, model, class_names, transform, model_name, topk, output_dir
        )

        if topk_probs is not None:
            results[image_file] = {
                'probabilities': topk_probs,
                'classes': topk_classes
            }

            # Salva os resultados no CSV
            if output_dir:
                with open(csv_path, 'a') as f:
                    row = [image_file]
                    for cls, prob in zip(topk_classes, topk_probs):
                        row.extend([cls, f"{prob * 100:.2f}"])
                    f.write(','.join(row) + '\n')

    return results


def process_directory_with_ensemble(images_dir, models, class_names, transforms_list, topk=3, output_dir=None):
    """
    Processa todas as imagens em um diretório usando um ensemble de modelos.

    Args:
        images_dir: Diretório contendo as imagens
        models: Lista de modelos para fazer a predição
        class_names: Nomes das classes
        transforms_list: Lista de transformações a serem aplicadas (uma para cada modelo)
        topk: Número de top-k predições a retornar
        output_dir: Diretório para salvar as visualizações

    Returns:
        results: Resultados das predições
    """
    # Verifica se o diretório existe
    if not os.path.exists(images_dir):
        print(f"❌ Diretório não encontrado: {images_dir}")
        return None

    # Obtém a lista de imagens
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_files = [f for f in os.listdir(images_dir)
                   if os.path.isfile(os.path.join(images_dir, f)) and
                   f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"❌ Nenhuma imagem encontrada em {images_dir}")
        return None

    print(f"📁 Processando {len(image_files)} imagens de {images_dir} com ensemble")

    # Cria o diretório de saída se necessário
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Resultados serão salvos em {output_dir}")

        # Cria um arquivo CSV para os resultados
        csv_path = os.path.join(output_dir, "ensemble_predictions.csv")
        with open(csv_path, 'w') as f:
            f.write("image,top1_class,top1_prob,top2_class,top2_prob,top3_class,top3_prob\n")

    # Processa cada imagem
    results = {}

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        print(f"[{i + 1}/{len(image_files)}] Processando {image_file}...")

        topk_probs, topk_classes = process_image_with_ensemble(
            image_path, models, class_names, transforms_list, topk, output_dir
        )

        if topk_probs is not None:
            results[image_file] = {
                'probabilities': topk_probs,
                'classes': topk_classes
            }

            # Salva os resultados no CSV
            if output_dir:
                with open(csv_path, 'a') as f:
                    row = [image_file]
                    for cls, prob in zip(topk_classes, topk_probs):
                        row.extend([cls, f"{prob * 100:.2f}"])
                    f.write(','.join(row) + '\n')

    return results


def main():
    """
    Função principal.
    """
    parser = argparse.ArgumentParser(description='Classificação de Serpentes com Ensemble')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', type=str, help='Caminho para um modelo único')
    group.add_argument('--ensemble', type=str, help='Caminho para o arquivo de configuração do ensemble')

    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument('--image', type=str, help='Caminho para uma imagem')
    group2.add_argument('--dir', type=str, help='Caminho para um diretório contendo imagens')

    parser.add_argument('--output', type=str, help='Diretório para salvar os resultados')
    parser.add_argument('--topk', type=int, default=3, help='Número de top-k predições (padrão: 3)')

    args = parser.parse_args()

    # Cria o diretório de saída se necessário
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Processa com um único modelo ou ensemble
    if args.model:
        model, class_names, transform, model_type = load_model(args.model)

        if args.image:
            process_image(args.image, model, class_names, transform, model_type, args.topk, args.output)
        else:
            process_directory(args.dir, model, class_names, transform, model_type, args.topk, args.output)
    else:
        models, class_names, transforms_list = load_ensemble(args.ensemble)

        if args.image:
            process_image_with_ensemble(args.image, models, class_names, transforms_list, args.topk, args.output)
        else:
            process_directory_with_ensemble(args.dir, models, class_names, transforms_list, args.topk, args.output)


if __name__ == '__main__':
    main()