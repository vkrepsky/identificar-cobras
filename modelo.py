import json
import shutil
import random
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights, VGG16_Weights, DenseNet121_Weights, Inception_V3_Weights, \
    ResNet50_Weights, EfficientNet_B3_Weights, EfficientNet_B0_Weights
from torch.cuda.amp import autocast, GradScaler
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.model_selection import train_test_split
from collections import Counter

# Otimizações CUDA - Melhoram o desempenho computacional na GPU
torch.backends.cudnn.benchmark = True  # Otimiza operações convolucionais, buscando os algoritmos mais eficientes
torch.backends.cudnn.enabled = True  # Garante que cuDNN seja usado quando disponível

# Configurações do modelo e treinamento
BATCH_SIZE = 64  # Tamanho do lote de imagens processadas em cada iteração
NUM_EPOCHS = 50  # Número total de épocas para treinamento do modelo
LEARNING_RATE = 0.001  # Taxa de aprendizado para o otimizador
IMAGE_SIZE = 224  # Tamanho da imagem para a rede (224x224 é padrão para muitas CNNs pré-treinadas)
NUM_WORKERS = 6  # Número de threads para carregamento paralelo de dados
DEVICE = torch.device("cuda")  # Define GPU como dispositivo de processamento
TARGET_IMAGES_PER_CLASS = 300  # Número desejado de imagens por classe após augmentation


class SnakeDataset(Dataset):
    """
    Classe personalizada para o dataset de serpentes que herda de torch.utils.data.Dataset
    Responsável por carregar as imagens e seus respectivos rótulos.
    """

    def __init__(self, image_paths, labels, transform=None):
        """
        Inicializa o dataset com caminhos de imagens, rótulos e transformações.

        Args:
            image_paths: Lista com os caminhos das imagens
            labels: Lista com os rótulos correspondentes a cada imagem
            transform: Transformações a serem aplicadas nas imagens
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Retorna o número total de imagens no dataset"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Carrega e retorna uma única amostra (imagem e rótulo).

        Args:
            idx: Índice da amostra a ser carregada

        Returns:
            Tupla contendo a imagem processada e seu rótulo
        """
        image = Image.open(self.image_paths[idx]).convert('RGB')  # Carrega imagem e converte para RGB
        if self.transform:
            image = self.transform(image)  # Aplica transformações na imagem
        return image, self.labels[idx]


class UNetEncoder(nn.Module):
    """Encoder da U-Net adaptado para classificação"""

    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(UNetEncoder, self).__init__()
        self.downs = nn.ModuleList()

        # Camadas de contração (encoder)
        for feature in features:
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )
            in_channels = feature

        # Feature pooling global
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Passa pelos blocos do encoder
        for down in self.downs:
            x = down(x)

        # Pooling global
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x


class SnakeClassifier(nn.Module):
    """
    Modelo de classificação de serpentes que suporta várias arquiteturas.
    Utiliza transfer learning com fine-tuning da última camada.
    """

    def __init__(self, num_classes, model_name='resnet18', dropout_rate=0.5):
        """
        Inicializa o modelo com a arquitetura base escolhida e adapta para a tarefa específica.

        Args:
            num_classes: Número de espécies de serpentes (classes) a classificar
            model_name: Nome da arquitetura base a ser usada
            dropout_rate: Taxa de dropout para regularização (valor entre 0 e 1)
        """
        super(SnakeClassifier, self).__init__()
        self.is_inception = False

        # Selecionar a arquitetura base com base no nome do modelo
        if model_name == 'resnet18':
            # ResNet18
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            num_features = self.model.fc.in_features

            # Congela os parâmetros da rede base para manter o conhecimento
            for param in self.model.parameters():
                param.requires_grad = False

            # Substitui a última camada para a classificação específica
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )

        elif model_name == 'vgg16':
            # VGG16
            self.model = models.vgg16(weights=VGG16_Weights.DEFAULT)
            num_features = self.model.classifier[6].in_features

            # Congela os parâmetros da rede base
            for param in self.model.parameters():
                param.requires_grad = False

            # Substitui a última camada do classificador
            self.model.classifier[6] = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )

        elif model_name == 'densenet121':
            # DenseNet121
            self.model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
            num_features = self.model.classifier.in_features

            # Congela os parâmetros da rede base
            for param in self.model.parameters():
                param.requires_grad = False

            # Substitui o classificador
            self.model.classifier = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )

        elif model_name == 'inception_v3':
            # Inception v3
            self.model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
            num_features = self.model.fc.in_features

            # Congela os parâmetros da rede base
            for param in self.model.parameters():
                param.requires_grad = False

            # Substitui o classificador principal
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )

            # Substitui o classificador auxiliar também
            self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, num_classes)

            # Flag para rastrear se este é o modelo Inception v3 (para tratamento especial no forward)
            self.is_inception = True

        elif model_name == 'yolo_classifier':
            # YOLO adaptado para classificação (usando darknet como inspiração)
            # Como YOLO é para detecção, usamos um modelo darknet-like inspirado no YOLO
            self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            num_features = self.model.fc.in_features

            # Congela os parâmetros da rede base
            for param in self.model.parameters():
                param.requires_grad = False

            # Substitui o classificador com estrutura inspirada em YOLO
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 1024),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )

        elif model_name == 'efficientnet_b0':
            # EfficientNet B0
            self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            num_features = self.model.classifier[1].in_features

            # Congela os parâmetros da rede base
            for param in self.model.parameters():
                param.requires_grad = False

            # Substitui o classificador
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )

        elif model_name == 'efficientnet_b3':
            # EfficientNet B3 (maior e mais preciso que B0)
            self.model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
            num_features = self.model.classifier[1].in_features

            # Congela os parâmetros da rede base
            for param in self.model.parameters():
                param.requires_grad = False

            # Substitui o classificador
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )

        elif model_name == 'unet':
            # U-Net adaptada para classificação
            # Usamos apenas o encoder da U-Net e adicionamos camadas de classificação
            encoder_channels = [64, 128, 256, 512]
            self.encoder = UNetEncoder(in_channels=3, features=encoder_channels)

            # Adiciona o classificador depois do encoder
            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
            )

            # Esta é uma abordagem personalizada, não usamos o model.forward padrão
            self.use_custom_forward = True

        else:
            raise ValueError(f"Modelo '{model_name}' não suportado")

    def forward(self, x):
        """
        Define o fluxo de dados através do modelo durante a fase forward.

        Args:
            x: Tensor de entrada (batch de imagens)

        Returns:
            Tensor de saída com as previsões do modelo
        """
        # Caso especial para U-Net adaptada
        if hasattr(self, 'use_custom_forward') and self.use_custom_forward:
            features = self.encoder(x)
            return self.classifier(features)

        # Tratamento especial para o modelo Inception v3
        elif self.is_inception:
            # Durante o treinamento, Inception v3 retorna tupla (saída principal, saída auxiliar)
            if self.training:
                output, aux_output = self.model(x)
                return output  # Durante o treinamento, ignoramos a saída auxiliar
            else:
                return self.model(x)  # No modo de avaliação, apenas retorna a saída principal
        else:
            return self.model(x)

    # Função auxiliar para compatibilidade com versões antigas do PyTorch
    def legacy_init(self, num_classes, model_name='resnet18', dropout_rate=0.5):
        """
        Inicialização alternativa para versões antigas do PyTorch (<1.13)
        que usam pretrained=True em vez de weights.

        Args:
            num_classes: Número de classes
            model_name: Nome do modelo
            dropout_rate: Taxa de dropout para regularização
        """
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
        elif model_name == 'densenet121':
            self.model = models.densenet121(pretrained=True)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
        elif model_name == 'inception_v3':
            self.model = models.inception_v3(pretrained=True, aux_logits=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
            self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, num_classes)
            self.is_inception = True
        elif model_name == 'yolo_classifier':
            self.model = models.resnet50(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 1024),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
        elif model_name == 'efficientnet_b0':
            try:
                self.model = models.efficientnet_b0(pretrained=True)
                num_features = self.model.classifier[1].in_features
                self.model.classifier = nn.Sequential(
                    nn.Dropout(p=dropout_rate, inplace=True),
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(512, num_classes)
                )
            except:
                raise ValueError("EfficientNet não disponível na sua versão do PyTorch. Atualize para 1.8+")
        else:
            raise ValueError(f"Modelo '{model_name}' não suportado em modo legado")

        # Congelar parâmetros
        for param in self.model.parameters():
            param.requires_grad = False

def calculate_class_weights(train_labels, class_names):
    """
    Calcula pesos para as classes com base na distribuição no conjunto de treinamento.

    Args:
        train_labels: Lista dos rótulos de treinamento
        class_names: Lista com os nomes das classes

    Returns:
        Tensor de pesos das classes para CrossEntropyLoss
    """
    print("\n🔍 Calculando pesos para balanceamento de classes...")

    # Contagem de cada classe
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    num_classes = len(class_names)

    # Verificar se todas as classes estão representadas
    for cls_idx in range(num_classes):
        if cls_idx not in class_counts:
            class_counts[cls_idx] = 0
            print(f"⚠️ Atenção: Classe {cls_idx} ({class_names[cls_idx]}) não tem exemplos no conjunto de treinamento!")

    # Calcular pesos usando fórmula de balanceamento
    class_weights = {}
    for cls_idx in range(num_classes):
        count = class_counts[cls_idx]
        if count == 0:
            # Atribuir um peso alto para classes sem exemplos
            class_weights[cls_idx] = 1.0
        else:
            # Peso inversamente proporcional à frequência da classe
            class_weights[cls_idx] = total_samples / (num_classes * count)

    # Normalizar pesos para ter média 1
    weight_sum = sum(class_weights.values())
    normalized_weights = {cls: (weight * num_classes / weight_sum) for cls, weight in class_weights.items()}

    # Criar tensor de pesos
    weight_tensor = torch.FloatTensor([normalized_weights[i] for i in range(num_classes)]).to(DEVICE)

    # Exibir informações sobre os pesos
    print("\nDistribuição de classes e pesos:")
    print(f"{'Classe':<20} {'Contagem':<10} {'Peso':<10} {'Peso Normalizado':<20}")
    print("-" * 60)
    for cls_idx in range(num_classes):
        class_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Classe {cls_idx}"
        cls_name_short = class_name[:18] + '..' if len(class_name) > 20 else class_name
        count = class_counts[cls_idx]
        weight = class_weights[cls_idx]
        norm_weight = normalized_weights[cls_idx]
        print(f"{cls_name_short:<20} {count:<10} {weight:.4f} {norm_weight:.4f}")

    return weight_tensor


def get_weighted_criterion(train_labels, class_names):
    """
    Cria uma função de perda ponderada para tratar o desbalanceamento de classes.

    Args:
        train_labels: Lista dos rótulos de treinamento
        class_names: Lista com os nomes das classes

    Returns:
        Função de perda CrossEntropyLoss com pesos
    """
    # Calcular pesos das classes
    class_weights = calculate_class_weights(train_labels, class_names)

    # Criar função de perda ponderada
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"\n✅ Usando CrossEntropyLoss com pesos para balanceamento de classes")

    return criterion

def check_augmented_directory(original_dir, augmented_dir, target_images_per_class, min_images_per_class):
    """
    Verifica se o diretório de imagens aumentadas já existe e contém dados suficientes.

    Args:
        original_dir: Diretório original com as imagens
        augmented_dir: Diretório onde deveriam estar as imagens aumentadas
        target_images_per_class: Número alvo de imagens por classe
        min_images_per_class: Número mínimo de imagens por classe

    Returns:
        bool: True se o diretório já existe e contém dados suficientes, False caso contrário
    """
    print(f"\n🔍 Verificando se diretório de imagens aumentadas já existe: {augmented_dir}")

    # Verifica se o diretório existe
    if not os.path.exists(augmented_dir):
        print(f"  ↳ Diretório não encontrado. Será necessário gerar as imagens aumentadas.")
        return False

    # Verifica as classes no diretório original
    original_classes = set()
    for item in os.listdir(original_dir):
        if os.path.isdir(os.path.join(original_dir, item)):
            class_images = [f for f in os.listdir(os.path.join(original_dir, item))
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            # Só considerar classes com o mínimo de imagens
            if len(class_images) >= min_images_per_class:
                original_classes.add(item)

    # Verifica as classes no diretório aumentado
    augmented_classes = set()
    augmented_class_counts = {}
    for item in os.listdir(augmented_dir):
        class_dir = os.path.join(augmented_dir, item)
        if os.path.isdir(class_dir):
            augmented_classes.add(item)
            # Conta as imagens desta classe
            class_images = [f for f in os.listdir(class_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            augmented_class_counts[item] = len(class_images)

    # Verifica se todas as classes originais estão presentes nas aumentadas
    missing_classes = original_classes - augmented_classes
    if missing_classes:
        print(f"  ↳ Faltam classes no diretório aumentado: {', '.join(missing_classes)}")
        return False

    # Verifica se cada classe tem imagens suficientes
    insufficient_classes = []
    for cls in original_classes:
        if cls in augmented_class_counts:
            if augmented_class_counts[cls] < target_images_per_class * 0.9:  # Tolerância de 90%
                insufficient_classes.append(f"{cls} ({augmented_class_counts[cls]} imagens)")

    if insufficient_classes:
        print(f"  ↳ Classes com imagens insuficientes: {', '.join(insufficient_classes)}")
        return False

    # Se chegou até aqui, o diretório está OK
    print(f"  ✅ Diretório de imagens aumentadas está completo!")
    print(f"  ↳ Total de {len(augmented_classes)} classes com aproximadamente {target_images_per_class} imagens cada")

    # Exibe detalhes das classes (opcional)
    print("\n📊 Estatísticas do diretório aumentado:")
    print(f"{'Classe':<25} {'Imagens':<10}")
    print("-" * 35)
    for cls in sorted(augmented_classes):
        if cls in augmented_class_counts:
            print(f"{cls:<25} {augmented_class_counts[cls]:<10}")

    return True


def augment_and_save_images(data_dir, output_dir, target_images_per_class):
    """
    Gera imagens aumentadas para cada classe até atingir o número alvo por classe
    e salva todas as imagens (originais + aumentadas) no diretório de saída.
    """
    print("\n" + "=" * 80)
    print("🔄 AUMENTAÇÃO DE DADOS - GERANDO IMAGENS SINTÉTICAS 🔄".center(80))
    print("=" * 80)

    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Transformações para data augmentation compatíveis com PIL Image
    pil_transforms = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    ]

    # Converter para tensor e depois para PIL para aplicar transformações que exigem tensor
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

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
                try:
                    # Começar com a imagem original
                    aug_img = img.copy()

                    # Aplicar transformações compatíveis com PIL
                    random.shuffle(pil_transforms)
                    num_transforms = random.randint(1, len(pil_transforms))

                    for t in range(num_transforms):
                        aug_img = pil_transforms[t](aug_img)

                    # Para transformações que exigem tensor (affine, perspective, blur):
                    # Vamos aplicar manualmente as transformações que seriam problemáticas
                    # Ao invés de usar as transformações originais que causam erro

                    # 1. Transformação affine manual (rotação, escala, etc.)
                    if random.random() > 0.5:
                        angle = random.uniform(-30, 30)
                        scale = random.uniform(0.8, 1.2)
                        aug_img = aug_img.rotate(angle, resample=Image.BICUBIC, expand=True)
                        width, height = aug_img.size
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        aug_img = aug_img.resize((new_width, new_height), Image.BICUBIC)

                    # 2. Simulação de blur usando resize
                    if random.random() > 0.7:
                        width, height = aug_img.size
                        # Reduz a resolução e depois aumenta para simular um blur
                        smaller = aug_img.resize((width // 4, height // 4), Image.BICUBIC)
                        aug_img = smaller.resize((width, height), Image.BICUBIC)

                    # Salvar imagem aumentada
                    output_path = os.path.join(output_class_dir, f"aug_{aug_idx}_{img_name}")
                    aug_img.save(output_path)
                    generated_count += 1

                except Exception as e:
                    print(f"  ⚠️ Erro geral ao processar {img_name}: {e}")
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

def get_data_transforms(model_name='resnet18'):
    """
    Define as transformações a serem aplicadas nas imagens para treinamento e validação.

    Args:
        model_name: Nome do modelo para ajustar o tamanho da imagem de entrada

    Returns:
        Tupla com (transformações_treino, transformações_validação)
    """
    # Ajusta o tamanho da imagem com base no modelo
    if model_name == 'inception_v3':
        image_size = 299  # Inception v3 espera 299x299
    else:
        image_size = IMAGE_SIZE  # 224x224 para outros modelos

    # Transformações para o conjunto de treinamento - inclui aumentação de dados
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Redimensiona as imagens
        transforms.RandomHorizontalFlip(),  # Espelha horizontalmente com 50% de chance
        transforms.RandomRotation(20),  # Rotação aleatória até 20 graus
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Variações de cor
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),  # Translações e escalas
        transforms.ToTensor(),  # Converte para tensor PyTorch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalização ImageNet
    ])

    # Transformações para o conjunto de validação - apenas redimensionamento e normalização
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Redimensiona as imagens
        transforms.ToTensor(),  # Converte para tensor PyTorch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalização ImageNet
    ])

    return train_transform, val_transform


def prepare_data(data_dir, min_images_per_class):
    """
    Prepara os dados para treinamento e validação, organizando caminhos e rótulos.
    Ignora classes com menos imagens que o mínimo especificado.

    Args:
        data_dir: Diretório contendo as imagens de serpentes organizadas por classe
        min_images_per_class: Número mínimo de imagens por classe para incluir no dataset

    Returns:
        Tupla com (caminhos_das_imagens, rótulos, nomes_das_classes, mapa_de_classes)
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

    return image_paths, labels, class_names, class_to_idx


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs,
                model_name='resnet18', patience=10, scheduler=None):
    """
    Função principal de treinamento do modelo com early stopping.

    Args:
        model: Modelo a ser treinado
        train_loader: DataLoader com dados de treinamento
        val_loader: DataLoader com dados de validação
        criterion: Função de perda
        optimizer: Otimizador para atualização dos pesos
        num_epochs: Número máximo de épocas para treinamento do modelo
        model_name: Nome do modelo para identificação nos logs e arquivos salvos
        patience: Número de épocas a esperar sem melhoria antes de parar (early stopping)
        scheduler: Opcional, scheduler para ajustar a taxa de aprendizado

    Returns:
        Melhor acurácia de validação alcançada
    """
    best_val_acc = 0.0
    best_val_loss = float('inf')
    scaler = GradScaler()

    # Variáveis para early stopping
    counter = 0
    early_stopping_triggered = False

    # Armazena métricas para comparação posterior
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    times_per_epoch = []

    # Verifica se é o modelo Inception v3
    is_inception = 'inception' in model_name.lower()

    # Ajusta tamanho de batch para Inception se necessário
    batch_size = 8 if is_inception else BATCH_SIZE

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # ----- Fase de Treinamento -----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        print(f"\n{'=' * 60}")
        print(f"Época [{epoch + 1}/{num_epochs}] - Iniciando treinamento com {model_name.upper()}")
        print(f"{'=' * 60}")

        # Itera sobre os lotes de treinamento
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            with autocast():
                # Forward pass
                if is_inception and model.training:
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2  # Ponderação da perda auxiliar
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

            # Atualiza os pesos do modelo
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # Gradient clipping para evitar explosão de gradientes
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # Acumula estatísticas
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            batch_correct = predicted.eq(labels).sum().item()
            train_correct += batch_correct

            # Força sincronização CUDA ocasionalmente
            if train_total % (10 * batch_size) == 0:
                torch.cuda.synchronize()

        # Calcula e exibe métricas de treinamento
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        print(f"\n{'>' * 20} Resultados de Treinamento - {model_name.upper()} {'<' * 20}")
        print(f"Perda média: {avg_train_loss:.4f}")
        print(f"Acurácia total: {train_acc:.2f}% ({train_correct}/{train_total} amostras corretas)")

        # ----- Fase de Validação -----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        print(f"\n{'=' * 60}")
        print(f"Época [{epoch + 1}/{num_epochs}] - Validação do modelo {model_name.upper()}")
        print(f"{'=' * 60}")

        with torch.no_grad(), autocast():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                batch_correct = predicted.eq(labels).sum().item()
                val_correct += batch_correct

                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
                    batch_acc = 100. * batch_correct / labels.size(0)
                    print(f"Lote [{batch_idx + 1}/{len(val_loader)}] | "
                          f"Perda: {loss.item():.4f} | "
                          f"Acurácia: {batch_acc:.2f}%")

        # Calcula e exibe métricas de validação
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        # Atualizar o scheduler com base na perda de validação
        if scheduler is not None:
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Taxa de aprendizado atual: {current_lr:.6f}")

        print(f"\n{'>' * 20} Resultados de Validação - {model_name.upper()} {'<' * 20}")
        print(f"Perda média: {avg_val_loss:.4f}")
        print(f"Acurácia total: {val_acc:.2f}% ({val_correct}/{val_total} amostras corretas)")

        # Calcula o tempo da época
        epoch_time = time.time() - epoch_start_time
        times_per_epoch.append(epoch_time)

        # Resumo da época
        print(f"\n{'#' * 60}")
        print(f"Resumo da Época {epoch + 1}/{num_epochs} - {model_name.upper()}:")
        print(f"Treino - Perda: {avg_train_loss:.4f}, Acurácia: {train_acc:.2f}%")
        print(f"Validação - Perda: {avg_val_loss:.4f}, Acurácia: {val_acc:.2f}%")
        print(f"Tempo de execução: {epoch_time:.2f}s")

        # Verifica se é o melhor modelo até agora e salva se for
        if val_acc > best_val_acc:
            melhoria = val_acc - best_val_acc
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            counter = 0  # Resetar contador de early stopping

            # Salva o estado do modelo, otimizador e métricas
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'model_name': model_name,
                'training_metrics': {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_accs': train_accs,
                    'val_accs': val_accs,
                    'times_per_epoch': times_per_epoch
                }
            }, f'best_model_{model_name}.pth')

            print(f"\n📊 NOVO MELHOR MODELO - {model_name.upper()}! 📊")
            print(f"Acurácia de validação melhorou em +{melhoria:.2f}%")
            print(f"Modelo salvo como 'best_model_{model_name}.pth'")
        else:
            # Incrementar contador de early stopping
            counter += 1
            print(f"\n⚠️ Sem melhoria na acurácia de validação há {counter} épocas.")

            # Verificar se early stopping deve ser acionado
            if counter >= patience:
                print(f"\n🛑 EARLY STOPPING: Nenhuma melhoria após {patience} épocas!")
                early_stopping_triggered = True
                break

        print(f"{'#' * 60}\n")

    # Exibir mensagem se early stopping for acionado
    if early_stopping_triggered:
        print(f"\n🔍 Early stopping foi acionado após {epoch + 1} épocas.")
        print(f"Melhor acurácia de validação: {best_val_acc:.2f}%")

    # Salva as métricas finais para este modelo
    model_metrics = {
        'model_name': model_name,
        'best_val_acc': best_val_acc,
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'times_per_epoch': times_per_epoch,
        'avg_epoch_time': sum(times_per_epoch) / len(times_per_epoch),
        'total_training_time': sum(times_per_epoch),
        'total_epochs': len(train_accs),
        'early_stopping_triggered': early_stopping_triggered
    }

    with open(f'metrics_{model_name}.json', 'w') as f:
        json.dump(model_metrics, f, indent=4)

    print(f"\n💾 Métricas de treinamento salvas em 'metrics_{model_name}.json'")

    return best_val_acc

def check_cuda_status():
    """
    Verifica e imprime informações sobre o ambiente CUDA/GPU.
    Útil para diagnóstico e debug.
    """
    print("\n=== DIAGNÓSTICO CUDA/GPU ===")
    print(f"PyTorch versão: {torch.__version__}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU atual: {torch.cuda.get_device_name()}")
        print(f"Número de GPUs: {torch.cuda.device_count()}")
        print(f"CUDA versão: {torch.version.cuda}")
        print(f"Memória GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        print(f"cuDNN habilitado: {torch.backends.cudnn.enabled}")
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")

        # Teste simples de operação na GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("\n✅ Teste de operação na GPU realizado com sucesso!")
    else:
        print("\n⚠️ ATENÇÃO: CUDA não está disponível! ⚠️")
        print("O modelo está rodando na CPU!")


def initialize_model_with_gradual_unfreezing(model, train_loader, num_epochs, model_name):
    """
    Implementa descongelamento gradual das camadas do modelo durante o treinamento.

    Args:
        model: O modelo inicializado
        train_loader: DataLoader para estimar número de passos
        num_epochs: Número total de épocas
        model_name: Nome do modelo

    Returns:
        Lista de callbacks para descongelar camadas em pontos específicos
    """
    # Calcular número total de passos para o treinamento
    total_steps = len(train_loader) * num_epochs

    # Implementar progressivamente para diferentes modelos
    if model_name == 'resnet18':
        # Inicialmente congelar todas as camadas exceto a última
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Lista de camadas a descongelar gradualmente
        layers_to_unfreeze = [
            ('layer4', int(total_steps * 0.2)),  # Descongelar layer4 após 20% do treinamento
            ('layer3', int(total_steps * 0.4)),  # Descongelar layer3 após 40% do treinamento
            ('layer2', int(total_steps * 0.6)),  # Descongelar layer2 após 60% do treinamento
            ('layer1', int(total_steps * 0.8)),  # Descongelar layer1 após 80% do treinamento
        ]

        print(f"\n🔄 Configurado descongelamento gradual para {model_name}:")
        for layer, step in layers_to_unfreeze:
            print(f"  → Camada {layer} será descongelada no passo {step}")

        return layers_to_unfreeze

    elif model_name == 'yolo_classifier':
        # Inicialmente, congelar toda a rede base
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Programar o descongelamento gradual (ajustar conforme a estrutura do modelo)
        layers_to_unfreeze = [
            ('layer4', int(total_steps * 0.2)),
            ('layer3', int(total_steps * 0.4)),
            ('layer2', int(total_steps * 0.6)),
            ('layer1', int(total_steps * 0.8)),
        ]

        print(f"\n🔄 Configurado descongelamento gradual para {model_name}:")
        for layer, step in layers_to_unfreeze:
            print(f"  → Camada {layer} será descongelada no passo {step}")

        return layers_to_unfreeze

    # Para outros modelos, retornar lista vazia (não implementado)
    print(f"\n⚠️ Descongelamento gradual não configurado para {model_name}")
    return []


def unfreeze_layers(model, current_step, layers_to_unfreeze, model_name):
    """
    Função para descongelar camadas específicas em um determinado passo.

    Args:
        model: O modelo sendo treinado
        current_step: Passo atual do treinamento
        layers_to_unfreeze: Lista de tuplas (nome_camada, passo_para_descongelar)
        model_name: Nome do modelo

    Returns:
        Boolean indicando se alguma camada foi descongelada
    """
    any_unfrozen = False

    for layer_name, step_to_unfreeze in layers_to_unfreeze:
        if current_step == step_to_unfreeze:
            # Descongelar camadas específicas com base no nome
            for name, param in model.named_parameters():
                if layer_name in name:
                    param.requires_grad = True

            print(f"\n🔓 DESCONGELANDO CAMADA: {layer_name} no passo {current_step}")
            any_unfrozen = True

            # Exibir status atual de camadas congeladas/descongeladas
            frozen_count = 0
            unfrozen_count = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    unfrozen_count += 1
                else:
                    frozen_count += 1

            print(f"  → Status atual: {unfrozen_count} parâmetros descongelados, {frozen_count} congelados")

    return any_unfrozen


def get_optimizer_and_scheduler(model, model_name):
    """
    Configura o otimizador com diferentes taxas de aprendizado para diferentes partes do modelo
    e um scheduler apropriado.

    Args:
        model: O modelo a ser treinado
        model_name: Nome do modelo

    Returns:
        Tupla (otimizador, scheduler)
    """
    # Parâmetros padrão
    lr = LEARNING_RATE
    weight_decay = 1e-4

    # Para modelos grandes, usar taxas de aprendizado diferentes para diferentes partes
    if model_name in ['vgg16', 'inception_v3', 'densenet121', 'efficientnet', 'unet_classifier']:
        # Grupos de parâmetros com diferentes taxas de aprendizado
        # 1. Camadas base (congeladas ou recém-descongeladas) - taxa menor
        # 2. Camadas intermediárias - taxa média
        # 3. Camadas finais (classificador) - taxa normal

        base_params = []
        mid_params = []
        classifier_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if 'fc' in name or 'classifier' in name:
                classifier_params.append(param)
            elif 'layer4' in name or 'Mixed_7' in name or 'denseblock4' in name:
                mid_params.append(param)
            else:
                base_params.append(param)

        param_groups = [
            {'params': base_params, 'lr': lr * 0.1, 'weight_decay': weight_decay},
            {'params': mid_params, 'lr': lr * 0.5, 'weight_decay': weight_decay},
            {'params': classifier_params, 'lr': lr, 'weight_decay': weight_decay * 0.5}
        ]

        # Usar SGD com momentum para modelos grandes
        optimizer = optim.SGD(param_groups, lr=lr, momentum=0.9, nesterov=True)

        print(f"\n⚙️ Usando SGD com diferentes taxas de aprendizado para {model_name}")
        print(f"  → Camadas base: lr={lr * 0.1}")
        print(f"  → Camadas intermediárias: lr={lr * 0.5}")
        print(f"  → Classificador: lr={lr}")
    else:
        # Para modelos menores, usar Adam com uma única taxa de aprendizado
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay
        )

        print(f"\n⚙️ Usando Adam com lr={lr} para {model_name}")

    # Configurar scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True,
        min_lr=lr * 0.01
    )

    print(f"📉 Scheduler: ReduceLROnPlateau (reduz lr pela metade após 5 épocas sem melhoria, min_lr={lr * 0.01})")

    return optimizer, scheduler


# Função para implementar acumulação de gradiente
def validate_model(model, val_loader, criterion, model_name, is_inception=False):
    """
    Valida o modelo no conjunto de validação.

    Args:
        model: Modelo a ser validado
        val_loader: DataLoader com dados de validação
        criterion: Função de perda
        model_name: Nome do modelo
        is_inception: Flag indicando se é o modelo Inception v3

    Returns:
        Tupla (perda_total, acurácia, num_corretos, total_amostras)
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    print(f"\n{'=' * 60}")
    print(f"Validação do modelo {model_name.upper()}")
    print(f"{'=' * 60}")

    with torch.no_grad(), autocast():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Acumula estatísticas
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            batch_correct = predicted.eq(labels).sum().item()
            val_correct += batch_correct

            # Mostra progresso ocasionalmente
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
                batch_acc = 100. * batch_correct / labels.size(0)
                print(f"Lote [{batch_idx + 1}/{len(val_loader)}] | "
                      f"Perda: {loss.item():.4f} | "
                      f"Acurácia: {batch_acc:.2f}%")

    # Calcula acurácia total
    val_acc = 100. * val_correct / val_total

    print(f"\n{'>' * 20} Resultados de Validação - {model_name.upper()} {'<' * 20}")
    print(f"Perda média: {val_loss / len(val_loader):.4f}")
    print(f"Acurácia total: {val_acc:.2f}% ({val_correct}/{val_total} amostras corretas)")

    return val_loss, val_acc, val_correct, val_total


def save_best_model(model, optimizer, best_val_acc, model_name, epoch,
                    train_losses, val_losses, train_accs, val_accs, times_per_epoch):
    """
    Salva o melhor modelo durante o treinamento.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'model_name': model_name,
        'training_metrics': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'times_per_epoch': times_per_epoch
        }
    }, f'best_model_{model_name}.pth')


def save_final_metrics(model_name, best_val_acc, train_accs, val_accs, train_losses,
                       val_losses, times_per_epoch, early_stopping_triggered):
    """
    Salva as métricas finais do treinamento.
    """
    model_metrics = {
        'model_name': model_name,
        'best_val_acc': best_val_acc,
        'final_train_acc': train_accs[-1] if train_accs else 0,
        'final_val_acc': val_accs[-1] if val_accs else 0,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'times_per_epoch': times_per_epoch,
        'avg_epoch_time': sum(times_per_epoch) / len(times_per_epoch) if times_per_epoch else 0,
        'total_training_time': sum(times_per_epoch) if times_per_epoch else 0,
        'total_epochs': len(train_accs),
        'early_stopping_triggered': early_stopping_triggered
    }

    with open(f'metrics_{model_name}.json', 'w') as f:
        json.dump(model_metrics, f, indent=4)

    print(f"\n💾 Métricas de treinamento salvas em 'metrics_{model_name}.json'")


def train_with_gradient_accumulation(model, train_loader, val_loader, criterion, optimizer,
                                     scheduler, num_epochs, model_name='resnet18',
                                     accumulation_steps=4, patience=10):
    """
    Função de treinamento com acumulação de gradiente para lidar com GPUs com menos memória.

    Args:
        model: Modelo a ser treinado
        train_loader: DataLoader com dados de treinamento
        val_loader: DataLoader com dados de validação
        criterion: Função de perda
        optimizer: Otimizador para atualização dos pesos
        scheduler: Scheduler para ajustar a taxa de aprendizado
        num_epochs: Número máximo de épocas para treinamento do modelo
        model_name: Nome do modelo para identificação nos logs e arquivos salvos
        accumulation_steps: Número de passos para acumular gradientes
        patience: Número de épocas a esperar sem melhoria antes de parar (early stopping)

    Returns:
        Melhor acurácia de validação alcançada
    """
    best_val_acc = 0.0
    best_val_loss = float('inf')
    scaler = GradScaler()

    # Variáveis para early stopping
    counter = 0
    early_stopping_triggered = False

    # Armazena métricas para comparação posterior
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    times_per_epoch = []

    # Verifica se é o modelo Inception v3
    is_inception = 'inception' in model_name.lower()
    is_unet = 'unet' in model_name.lower()

    # Configurar descongelamento gradual
    layers_to_unfreeze = initialize_model_with_gradual_unfreezing(model, train_loader, num_epochs, model_name)

    # Contador global de passos para descongelamento gradual
    global_step = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # ----- Fase de Treinamento -----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        print(f"\n{'=' * 60}")
        print(f"Época [{epoch + 1}/{num_epochs}] - Iniciando treinamento com {model_name.upper()}")
        print(f"{'=' * 60}")

        # Resetar o otimizador no início de cada época
        optimizer.zero_grad(set_to_none=True)

        # Itera sobre os lotes de treinamento
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Verificar e descongelar camadas se necessário
            unfreeze_layers(model, global_step, layers_to_unfreeze, model_name)

            with autocast():
                # Forward pass
                if is_inception and model.training:
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                # Normalizar a perda pelo número de passos de acumulação
                loss = loss / accumulation_steps

            # Propagar o gradiente
            scaler.scale(loss).backward()

            # Acumular estatísticas
            train_loss += loss.item() * accumulation_steps  # Multiplicar pela acumulação para estatísticas corretas
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            batch_correct = predicted.eq(labels).sum().item()
            train_correct += batch_correct

            # Incrementar o contador global de passos
            global_step += 1

            # Atualizar pesos após acumular gradientes por vários passos
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping para evitar explosão de gradientes
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Atualizar pesos
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # Exibir progresso
                if (batch_idx + 1) % (10 * accumulation_steps) == 0:
                    processed = batch_idx * len(inputs)
                    total = len(train_loader.dataset)
                    batch_acc = 100. * batch_correct / len(inputs)
                    print(f"Treinamento: [{processed}/{total} ({100. * processed / total:.1f}%)] "
                          f"Perda: {loss.item() * accumulation_steps:.4f} Acurácia: {batch_acc:.2f}%")

        # Calcula e exibe métricas de treinamento
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        print(f"\n{'>' * 20} Resultados de Treinamento - {model_name.upper()} {'<' * 20}")
        print(f"Perda média: {avg_train_loss:.4f}")
        print(f"Acurácia total: {train_acc:.2f}% ({train_correct}/{train_total} amostras corretas)")

        # ----- Fase de Validação -----
        val_loss, val_acc, val_correct, val_total = validate_model(
            model, val_loader, criterion, model_name, is_inception)

        # Atualizar métricas de validação
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        # Atualizar o scheduler com base na perda de validação
        if scheduler is not None:
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Taxa de aprendizado atual: {current_lr:.6f}")

        # Calcula o tempo da época
        epoch_time = time.time() - epoch_start_time
        times_per_epoch.append(epoch_time)

        # Resumo da época
        print(f"\n{'#' * 60}")
        print(f"Resumo da Época {epoch + 1}/{num_epochs} - {model_name.upper()}:")
        print(f"Treino - Perda: {avg_train_loss:.4f}, Acurácia: {train_acc:.2f}%")
        print(f"Validação - Perda: {avg_val_loss:.4f}, Acurácia: {val_acc:.2f}%")
        print(f"Tempo de execução: {epoch_time:.2f}s")

        # Verifica se é o melhor modelo até agora e salva se for
        if val_acc > best_val_acc:
            melhoria = val_acc - best_val_acc
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            counter = 0  # Resetar contador de early stopping

            # Salva o estado do modelo, otimizador e métricas
            save_best_model(model, optimizer, best_val_acc, model_name, epoch,
                            train_losses, val_losses, train_accs, val_accs, times_per_epoch)

            print(f"\n📊 NOVO MELHOR MODELO - {model_name.upper()}! 📊")
            print(f"Acurácia de validação melhorou em +{melhoria:.2f}%")
        else:
            # Incrementar contador de early stopping
            counter += 1
            print(f"\n⚠️ Sem melhoria na acurácia de validação há {counter} épocas.")

            # Verificar se early stopping deve ser acionado
            if counter >= patience:
                print(f"\n🛑 EARLY STOPPING: Nenhuma melhoria após {patience} épocas!")
                early_stopping_triggered = True
                break

        print(f"{'#' * 60}\n")

    # Exibir mensagem se early stopping for acionado
    if early_stopping_triggered:
        print(f"\n🔍 Early stopping foi acionado após {epoch + 1} épocas.")
        print(f"Melhor acurácia de validação: {best_val_acc:.2f}%")

    # Salvar métricas finais
    save_final_metrics(model_name, best_val_acc, train_accs, val_accs, train_losses,
                       val_losses, times_per_epoch, early_stopping_triggered)

    return best_val_acc

def parse_arguments():
    """
    Processa argumentos da linha de comando.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Treinamento de classificador de serpentes com múltiplos modelos')
    parser.add_argument('--data-dir', type=str, default=r"C:\tcc\imgs-inaturalist",
                        help='Diretório contendo as imagens organizadas por classe')
    parser.add_argument('--augmented-dir', type=str, default=r"C:\tcc\imgs-inaturalist_augmented",
                        help='Diretório onde salvar as imagens aumentadas')
    parser.add_argument('--target-per-class', type=int, default=300,
                        help='Número alvo de imagens por classe após a aumentação')
    parser.add_argument('--min-images', type=int, default=10,
                        help='Número mínimo de imagens por classe para incluir no treinamento')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Tamanho do batch para treinamento')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Número de épocas de treinamento')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Taxa de aprendizado para o otimizador')
    parser.add_argument('--workers', type=int, default=4,
                        help='Número de workers para carregamento de dados')
    parser.add_argument('--skip-augmentation', action='store_true',
                        help='Pular a etapa de aumentação de dados')
    parser.add_argument('--single-model', action='store_true',
                        help='Treinar apenas um modelo em vez de comparar todos')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'vgg16', 'densenet121', 'inception_v3', 'yolo_classifier'],
                        help='Modelo a ser usado quando --single-model está ativado')
    parser.add_argument('--force-augmentation', action='store_true',
                        help='Forçar a regeneração das imagens aumentadas mesmo se o diretório já existir')
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Continuar a comparação a partir de um modelo específico')

    args = parser.parse_args()

    return args


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import pandas as pd
import os


def plot_learning_curves(model_name):
    """
    Plota as curvas de aprendizado (acurácia e perda) para treinamento e validação
    para analisar possíveis sinais de overfitting.

    Args:
        model_name: Nome do modelo para carregar as métricas salvas
    """
    try:
        # Carrega as métricas salvas
        with open(f'metrics_{model_name}.json', 'r') as f:
            metrics = json.load(f)

        # Extrai os dados
        train_losses = metrics['train_losses']
        val_losses = metrics['val_losses']
        train_accs = metrics['train_accs']
        val_accs = metrics['val_accs']
        epochs = range(1, len(train_losses) + 1)

        # Configuração da figura
        plt.figure(figsize=(15, 6))

        # Plot de acurácia
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_accs, 'bo-', label='Acurácia de Treinamento')
        plt.plot(epochs, val_accs, 'ro-', label='Acurácia de Validação')
        plt.title(f'Curvas de Acurácia - {model_name}')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia (%)')
        plt.legend()
        plt.grid(True)

        # Destacar a diferença entre as curvas (possível overfitting)
        plt.fill_between(epochs, train_accs, val_accs, alpha=0.2, color='yellow',
                         label='Gap Treino-Validação')

        # Plot de perda
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_losses, 'bo-', label='Perda de Treinamento')
        plt.plot(epochs, val_losses, 'ro-', label='Perda de Validação')
        plt.title(f'Curvas de Perda - {model_name}')
        plt.xlabel('Épocas')
        plt.ylabel('Perda')
        plt.legend()
        plt.grid(True)

        # Destacar a diferença entre as curvas (possível overfitting)
        plt.fill_between(epochs, train_losses, val_losses, alpha=0.2, color='yellow',
                         label='Gap Treino-Validação')

        # Salvar o gráfico
        plt.tight_layout()
        plt.savefig(f'learning_curves_{model_name}.png')
        plt.close()

        print(f"✅ Curvas de aprendizado salvas em 'learning_curves_{model_name}.png'")

        # Analisar sinais de overfitting
        last_5_train_acc = train_accs[-5:]
        last_5_val_acc = val_accs[-5:]
        avg_gap = sum([t - v for t, v in zip(last_5_train_acc, last_5_val_acc)]) / 5

        # Análise do gap entre treinamento e validação
        print("\n📊 ANÁLISE DE OVERFITTING:")
        print(f"Gap médio nas últimas 5 épocas: {avg_gap:.2f}%")

        if avg_gap > 15:
            print("⚠️ ALERTA DE OVERFITTING: Gap significativo entre treino e validação (>15%).")
            print("   Recomendações: Aumentar regularização, dropout ou reduzir complexidade do modelo.")
        elif avg_gap > 7:
            print("⚠️ POSSÍVEL OVERFITTING: Gap moderado entre treino e validação (>7%).")
            print("   Monitore de perto e considere técnicas de regularização adicional.")
        else:
            print("✅ SEM SINAIS DE OVERFITTING: Gap pequeno entre treino e validação.")
            print("   O modelo aparenta estar generalizando bem.")

        # Verificar se a validação parou de melhorar
        peak_val_acc = max(val_accs)
        peak_epoch = val_accs.index(peak_val_acc) + 1
        epochs_since_improvement = len(val_accs) - peak_epoch

        if epochs_since_improvement > 5:
            print(f"\n⚠️ ALERTA: A acurácia de validação não melhorou nos últimos {epochs_since_improvement} épocas.")
            print(f"   Melhor acurácia de validação: {peak_val_acc:.2f}% (atingida na época {peak_epoch})")
            print("   Considere usar early stopping em treinos futuros.")
        else:
            print(f"\n✅ O modelo atingiu sua melhor acurácia de validação ({peak_val_acc:.2f}%) na época {peak_epoch}.")

        return {
            'avg_gap': avg_gap,
            'peak_val_acc': peak_val_acc,
            'peak_epoch': peak_epoch,
            'epochs_since_improvement': epochs_since_improvement,
            'has_overfitting': avg_gap > 15
        }

    except Exception as e:
        print(f"❌ Erro ao plotar curvas de aprendizado: {e}")
        return None


def generate_confusion_matrix(model, data_loader, class_names, model_name):
    """
    Gera uma matriz de confusão para avaliar o desempenho do modelo por classe.

    Args:
        model: Modelo treinado
        data_loader: DataLoader com dados de validação
        class_names: Lista com os nomes das classes
        model_name: Nome do modelo para salvar o arquivo
    """
    try:
        # Coleta previsões e rótulos verdadeiros
        y_true = []
        y_pred = []

        model.eval()
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

        # Calcula a matriz de confusão
        cm = confusion_matrix(y_true, y_pred)

        # Normaliza a matriz
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plota a matriz de confusão
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Matriz de Confusão Normalizada - {model_name}')
        plt.ylabel('Classe Verdadeira')
        plt.xlabel('Classe Prevista')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name}.png')
        plt.close()

        print(f"✅ Matriz de confusão salva em 'confusion_matrix_{model_name}.png'")

        # Análise da matriz de confusão para identificar classes problemáticas
        accuracy_per_class = cm_normalized.diagonal()
        worst_classes = [(class_names[i], accuracy_per_class[i]) for i in np.argsort(accuracy_per_class)[:3]]
        best_classes = [(class_names[i], accuracy_per_class[i]) for i in np.argsort(accuracy_per_class)[-3:]]

        print("\n📊 ANÁLISE POR CLASSE:")
        print("Classes com maior precisão:")
        for cls, acc in best_classes[::-1]:
            print(f"   {cls}: {acc * 100:.2f}%")

        print("Classes com menor precisão:")
        for cls, acc in worst_classes:
            print(f"   {cls}: {acc * 100:.2f}%")

        # Gera relatório de classificação
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Salva o relatório
        report_df.to_csv(f'classification_report_{model_name}.csv')
        print(f"✅ Relatório de classificação salvo em 'classification_report_{model_name}.csv'")

        return {
            'worst_classes': worst_classes,
            'best_classes': best_classes,
            'report': report
        }

    except Exception as e:
        print(f"❌ Erro ao gerar matriz de confusão: {e}")
        return None


def analyze_model_performance(model_name, model, val_loader, class_names):
    """
    Realiza uma análise completa do desempenho do modelo, incluindo:
    - Curvas de aprendizado
    - Matriz de confusão
    - Análise de overfitting

    Args:
        model_name: Nome do modelo
        model: O modelo treinado
        val_loader: DataLoader do conjunto de validação
        class_names: Lista de nomes das classes

    Returns:
        Um dicionário com os resultados da análise
    """
    print(f"\n{'=' * 80}")
    print(f"📊 ANÁLISE DE DESEMPENHO DO MODELO: {model_name.upper()} 📊".center(80))
    print(f"{'=' * 80}")

    # Análise das curvas de aprendizado
    learning_curves_analysis = plot_learning_curves(model_name)

    # Geração da matriz de confusão
    confusion_matrix_analysis = generate_confusion_matrix(model, val_loader, class_names, model_name)

    # Resultados consolidados
    results = {
        'model_name': model_name,
        'learning_curves': learning_curves_analysis,
        'confusion_matrix': confusion_matrix_analysis,
    }

    # Salva os resultados completos para referência
    with open(f'performance_analysis_{model_name}.json', 'w') as f:
        # Converta objetos não-serializáveis em strings
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_dict = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_dict[k] = v.tolist()
                    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], tuple):
                        serializable_dict[k] = [[t[0], float(t[1])] for t in v]
                    else:
                        serializable_dict[k] = v
                serializable_results[key] = serializable_dict
            else:
                serializable_results[key] = value

        json.dump(serializable_results, f, indent=4)

    print(f"\n✅ Análise de desempenho completa salva em 'performance_analysis_{model_name}.json'")

    # Recomendações finais
    print("\n📋 RECOMENDAÇÕES FINAIS:")

    if learning_curves_analysis and learning_curves_analysis.get('has_overfitting', False):
        print("⚠️ O modelo apresenta sinais de overfitting. Considere:")
        print("   - Aumentar o dropout nas camadas finais")
        print("   - Aplicar regularização L2 mais forte")
        print("   - Reduzir a complexidade do modelo")
        print("   - Aumentar o conjunto de dados de treinamento")
    else:
        print("✅ O modelo parece estar generalizando bem.")

    if confusion_matrix_analysis and confusion_matrix_analysis.get('worst_classes'):
        print("\n⚠️ Classes problemáticas identificadas. Considere:")
        print("   - Aumentar os dados de treinamento para essas classes específicas")
        print("   - Usar técnicas de data augmentation mais agressivas para essas classes")
        print("   - Investigar se há problemas de qualidade nos dados dessas classes")

    print("\n" + "=" * 80)
    return results


# Função para adicionar ao main() após treinar cada modelo
def evaluate_trained_model(model, val_loader, class_names, model_name):
    # Carrega o melhor modelo salvo
    checkpoint = torch.load(f'best_model_{model_name}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Realiza análise completa
    analysis_results = analyze_model_performance(model_name, model, val_loader, class_names)

    return analysis_results


def main():
    """
    Função principal atualizada que orquestra todo o processo de treinamento
    com as modificações anti-overfitting.
    """
    # Processa argumentos da linha de comando
    args = parse_arguments()

    # Atualiza configurações globais com base nos argumentos
    global BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, NUM_WORKERS
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    NUM_WORKERS = args.workers

    print("\n" + "=" * 80)
    print("🐍 INICIANDO TREINAMENTO ANTI-OVERFITTING DO CLASSIFICADOR DE SERPENTES 🐍".center(80))
    print("=" * 80 + "\n")

    # Verifica configuração da GPU
    check_cuda_status()

    # Diretório das imagens originais e aumentadas
    data_dir = args.data_dir
    augmented_dir = args.augmented_dir
    target_per_class = args.target_per_class
    min_images_per_class = args.min_images

    # Etapa de aumentação de dados (opcional)
    if not args.skip_augmentation:
        try:
            # Verifica se o diretório já existe e está completo
            if args.force_augmentation or not check_augmented_directory(data_dir, augmented_dir, target_per_class,
                                                                        min_images_per_class):
                print(f"\n🔍 Aumentando dados para atingir {target_per_class} imagens por classe...")
                augmented_dir = augment_and_save_images(data_dir, augmented_dir, target_per_class)
            else:
                print(f"\n✅ Usando diretório de imagens aumentadas existente: {augmented_dir}")

            # Usar o diretório aumentado para treinamento
            data_dir = augmented_dir
        except Exception as e:
            print(f"\n⚠️ Aviso: Erro durante a etapa de aumentação de dados: {e}")
            import traceback
            traceback.print_exc()
            print(f"Continuando com as imagens originais em: {data_dir}")
    else:
        print(f"\n⚠️ Etapa de aumentação de dados pulada. Usando diretório original: {data_dir}")

    # Prepara os dados, ignorando classes com menos do que o número mínimo de imagens
    try:
        image_paths, labels, class_names, class_to_idx = prepare_data(data_dir, min_images_per_class)
    except ValueError as e:
        print(f"\n❌ ERRO: {e}")
        print(f"Tente ajustar o parâmetro --min-images para um valor menor ou adicionar mais imagens.")
        return

    # Divide os dados em conjuntos de treinamento e validação (80% / 20%)
    print("\n📊 Dividindo dados em conjuntos de treinamento (80%) e validação (20%)...")
    try:
        # MODIFICADO: Usar stratify para garantir distribuição equilibrada
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print(f"Conjunto de treinamento: {len(train_paths)} imagens")
        print(f"Conjunto de validação: {len(val_paths)} imagens")
    except ValueError as e:
        print(f"\n⚠️ Aviso: Não foi possível usar amostragem estratificada: {e}")
        # Fallback para divisão não estratificada
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=None
        )
        print(f"Usando divisão não estratificada.")
        print(f"Conjunto de treinamento: {len(train_paths)} imagens")
        print(f"Conjunto de validação: {len(val_paths)} imagens")

    # Por padrão, compara todos os modelos a menos que --single-model seja especificado
    if not args.single_model:
        # MODIFICADO: Adicionados novos modelos
        models_to_compare = ['resnet18', 'vgg16', 'densenet121', 'inception_v3',
                             'yolo_classifier', 'efficientnet', 'unet_classifier']
        model_results = {}

        print("\n" + "=" * 80)
        print("🔄 COMPARAÇÃO DE MÚLTIPLOS MODELOS DE CLASSIFICAÇÃO (ANTI-OVERFITTING) 🔄".center(80))
        print("=" * 80)

        # Data e hora do início da comparação para relatório
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Se continuar de um modelo específico
        start_index = 0
        if args.continue_from and args.continue_from in models_to_compare:
            start_index = models_to_compare.index(args.continue_from)
            print(f"\n⚠️ Continuando comparação a partir do modelo {args.continue_from}")

            # Carrega resultados anteriores se existirem
            try:
                with open('model_comparison_report.json', 'r') as f:
                    previous_report = json.load(f)
                    if 'results' in previous_report:
                        for model, result in previous_report['results'].items():
                            if models_to_compare.index(model) < start_index:
                                model_results[model] = result
                                print(
                                    f"Carregou resultado anterior: {model} - Acurácia: {result.get('accuracy', 'N/A')}")
            except (FileNotFoundError, json.JSONDecodeError):
                print("Nenhum relatório anterior encontrado ou erro ao carregar. Iniciando novos resultados.")

        # Iterar pelos modelos
        total_models = len(models_to_compare)
        for i, model_name in enumerate(models_to_compare[start_index:], start=start_index + 1):
            print(f"\n\n{'=' * 80}")
            print(f"🔍 TREINANDO MODELO {i}/{total_models}: {model_name.upper()} (ANTI-OVERFITTING) 🔍".center(80))
            print(f"{'=' * 80}\n")

            # Obtém transformações específicas para este modelo
            print(f"\n🔄 Configurando transformações de imagem para {model_name}...")
            train_transform, val_transform = get_data_transforms(model_name)

            # Cria os datasets com as transformações específicas
            train_dataset = SnakeDataset(train_paths, train_labels, train_transform)
            val_dataset = SnakeDataset(val_paths, val_labels, val_transform)

            # Cria DataLoaders
            print(f"\n⚙️ Configurando DataLoaders para {model_name}...")
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )

            # Inicializa o modelo
            num_classes = len(class_names)
            print(f"\n🧠 Inicializando modelo {model_name} para classificar {num_classes} espécies de serpentes...")

            try:
                # Libera memória CUDA antes de criar um novo modelo
                torch.cuda.empty_cache()

                # MODIFICADO: Criar modelo com dropout aumentado
                model = SnakeClassifier(num_classes, model_name=model_name, dropout_rate=0.7).to(DEVICE)

                # MODIFICADO: Criar criterion com balanceamento de classes
                criterion = get_weighted_criterion(train_labels, class_names)

                # MODIFICADO: Configurar otimizador e scheduler otimizados
                optimizer, scheduler = get_optimizer_and_scheduler(model, model_name)

                # Treina o modelo
                print(f"\n🚀 Iniciando treinamento do modelo {model_name} com técnicas anti-overfitting...")
                print(f"Épocas: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}")

                # Inicia o cronômetro para este modelo
                model_start_time = time.time()

                # MODIFICADO: Treina o modelo com acumulação de gradiente e early stopping
                best_val_acc = train_with_gradient_accumulation(
                    model, train_loader, val_loader, criterion, optimizer, scheduler,
                    NUM_EPOCHS, model_name=model_name, accumulation_steps=4, patience=10
                )

                # Calcula tempo total de treinamento
                model_train_time = time.time() - model_start_time

                # Salva o resultado
                model_results[model_name] = {
                    'accuracy': best_val_acc,
                    'training_time': model_train_time
                }

                print(f"\n✅ Treinamento do modelo {model_name} concluído com sucesso!")
                print(f"Melhor acurácia de validação: {best_val_acc:.2f}%")
                print(f"Tempo de treinamento: {model_train_time:.2f}s ({model_train_time / 60:.2f}min)")

                # Salva os nomes das classes e seu mapeamento para uso futuro
                class_info = {
                    "model_name": model_name,
                    "class_names": class_names,
                    "class_to_idx": class_to_idx,
                    "best_validation_accuracy": best_val_acc,
                    "training_time_seconds": model_train_time,
                    "training_config": {
                        "min_images_per_class": min_images_per_class,
                        "batch_size": BATCH_SIZE,
                        "epochs": NUM_EPOCHS,
                        "learning_rate": LEARNING_RATE,
                        "data_augmentation": not args.skip_augmentation,
                        "target_images_per_class": target_per_class if not args.skip_augmentation else None,
                        "anti_overfitting": True,
                        "class_balanced_loss": True,
                        "early_stopping": True
                    }
                }

                with open(f'class_info_{model_name}.json', 'w') as f:
                    json.dump(class_info, f, indent=4)

                # MODIFICADO: Realiza análise de desempenho no conjunto de validação
                evaluate_trained_model(model, val_loader, class_names, model_name)

                # Salva o relatório parcial após cada modelo para permitir retomada
                interim_report = {
                    "start_time": start_time,
                    "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "models_compared": models_to_compare,
                    "current_model_index": i,
                    "results": model_results,
                    "dataset_info": {
                        "data_dir": data_dir,
                        "num_classes": len(class_names),
                        "total_images": len(image_paths),
                        "train_images": len(train_paths),
                        "val_images": len(val_paths)
                    },
                    "training_config": {
                        "batch_size": BATCH_SIZE,
                        "epochs": NUM_EPOCHS,
                        "learning_rate": LEARNING_RATE,
                        "anti_overfitting": True
                    }
                }

                with open('model_comparison_report.json', 'w') as f:
                    json.dump(interim_report, f, indent=4)

                print(f"\n💾 Relatório parcial salvo.")

                next_model = models_to_compare[i] if i < len(models_to_compare) else None
                if next_model:
                    print(f"Próximo modelo: {next_model}")
                    print(f"Para retomar posteriormente, use:\npython modelo.py --continue-from={next_model}")

            except Exception as e:
                print(f"\n❌ ERRO ao treinar o modelo {model_name}: {e}")
                import traceback
                traceback.print_exc()

                model_results[model_name] = {'accuracy': "Falha", 'training_time': 0, 'error': str(e)}

                # Salva relatório mesmo em caso de falha
                error_report = {
                    "start_time": start_time,
                    "error_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "models_compared": models_to_compare,
                    "failed_model": model_name,
                    "error_message": str(e),
                    "results": model_results
                }

                with open('model_comparison_error_report.json', 'w') as f:
                    json.dump(error_report, f, indent=4)

                print(f"\n⚠️ Erro salvo em model_comparison_error_report.json")
                print(
                    f"Para continuar a partir do próximo modelo, use --continue-from={models_to_compare[i] if i < len(models_to_compare) - 1 else 'N/A'}")

                # Perguntar se deseja continuar após um erro
                if i < len(models_to_compare) - 1:
                    try:
                        cont = input("\nDeseja continuar para o próximo modelo? (s/n): ")
                        if cont.lower() != 's':
                            print("Interrompendo a comparação de modelos.")
                            break
                    except:
                        # Em caso de execução não interativa, continua automaticamente
                        print("Modo não interativo detectado, continuando para o próximo modelo...")

        # Exibe comparação final
        print("\n" + "=" * 80)
        print("📊 COMPARAÇÃO FINAL DOS MODELOS (ANTI-OVERFITTING) 📊".center(80))
        print("=" * 80)
        print(f"{'Modelo':<20} {'Melhor Acurácia':<20} {'Tempo de Treinamento':<25}")
        print("-" * 65)

        for model_name, result in model_results.items():
            if result['accuracy'] != "Falha":
                print(
                    f"{model_name:<20} {result['accuracy']:.2f}% {result['training_time']:.2f}s ({result['training_time'] / 60:.2f}min)")
            else:
                print(f"{model_name:<20} {result['accuracy']} N/A")

        # Identifica o melhor modelo
        best_model = max(
            [(name, res) for name, res in model_results.items() if res['accuracy'] != "Falha"],
            key=lambda x: x[1]['accuracy'],
            default=(None, {'accuracy': 0})
        )

        if best_model[0]:
            print("\n" + "-" * 65)
            print(f"🏆 MELHOR MODELO: {best_model[0]} (Acurácia: {best_model[1]['accuracy']:.2f}%)")

        # Salva relatório de comparação
        comparison_report = {
            "start_time": start_time,
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models_compared": models_to_compare,
            "results": model_results,
            "best_model": best_model[0],
            "best_accuracy": best_model[1]['accuracy'] if best_model[0] else "N/A",
            "dataset_info": {
                "data_dir": data_dir,
                "num_classes": len(class_names),
                "total_images": len(image_paths),
                "train_images": len(train_paths),
                "val_images": len(val_paths)
            },
            "training_config": {
                "batch_size": BATCH_SIZE,
                "epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "anti_overfitting": True
            }
        }

        with open('model_comparison_antioverfitting_report.json', 'w') as f:
            json.dump(comparison_report, f, indent=4)

        print("\n✅ Comparação de modelos concluída!")
        print("Relatório de comparação salvo em 'model_comparison_antioverfitting_report.json'")

    else:
        # Treinamento de um único modelo com modificações anti-overfitting
        model_name = args.model

        # Obtém transformações para os dados
        print("\n🔄 Configurando transformações de imagem e aumentação de dados...")
        train_transform, val_transform = get_data_transforms(model_name)

        # Cria os datasets
        train_dataset = SnakeDataset(train_paths, train_labels, train_transform)
        val_dataset = SnakeDataset(val_paths, val_labels, val_transform)

        # Cria DataLoaders otimizados
        print("\n⚙️ Configurando DataLoaders...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        # Inicializa o modelo com anti-overfitting
        num_classes = len(class_names)
        print(
            f"\n🧠 Inicializando modelo {model_name} anti-overfitting para classificar {num_classes} espécies de serpentes...")
        model = SnakeClassifier(num_classes, model_name=model_name, dropout_rate=0.7).to(DEVICE)

        # Define função de perda com balanceamento de classes
        criterion = get_weighted_criterion(train_labels, class_names)

        # Configurar otimizador e scheduler otimizados
        optimizer, scheduler = get_optimizer_and_scheduler(model, model_name)

        # Treina o modelo
        print("\n🚀 Iniciando treinamento do modelo com técnicas anti-overfitting...")
        print(f"Épocas: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}")
        best_acc = train_with_gradient_accumulation(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            NUM_EPOCHS, model_name=model_name, accumulation_steps=4, patience=10
        )

        # Salva os nomes das classes e seu mapeamento para uso futuro
        print("\n💾 Salvando nomes das classes e mapeamento para inferência...")
        class_info = {
            "model_name": model_name,
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "best_validation_accuracy": best_acc,
            "training_config": {
                "min_images_per_class": min_images_per_class,
                "batch_size": BATCH_SIZE,
                "epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "data_augmentation": not args.skip_augmentation,
                "target_images_per_class": target_per_class if not args.skip_augmentation else None,
                "anti_overfitting": True
            }
        }

        with open(f'class_info_{model_name}_antioverfitting.json', 'w') as f:
            json.dump(class_info, f, indent=4)

        # Realiza análise de desempenho
        evaluate_trained_model(model, val_loader, class_names, model_name)

        print("\n✅ Treinamento concluído com sucesso!")
        print(f"Modelo salvo: 'best_model_{model_name}.pth'")
        print(f"Informações das classes salvas: 'class_info_{model_name}_antioverfitting.json'")
        print(f"Melhor acurácia obtida: {best_acc:.2f}%")

    print("\n👋 Obrigado por usar o Classificador de Serpentes Anti-Overfitting!")


if __name__ == "__main__":
    main()