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

# Diretórios para salvar resultados
MODELS_DIR = "models"  # Para modelos treinados
PLOTS_DIR = "plots"  # Para gráficos
REPORTS_DIR = "reports"  # Para relatórios e métricas
AUGMENTED_DIR = "augmented"  # Para imagens aumentadas

# Otimizações CUDA - Melhoram o desempenho computacional na GPU
torch.backends.cudnn.benchmark = True  # Otimiza operações convolucionais, buscando os algoritmos mais eficientes
torch.backends.cudnn.enabled = True  # Garante que cuDNN seja usado quando disponível

# Configurações do modelo e treinamento
BATCH_SIZE = 64  # Tamanho do lote de imagens processadas em cada iteração
NUM_EPOCHS = 300
# Número total de épocas para treinamento do modelo
LEARNING_RATE = 0.001  # Taxa de aprendizado para o otimizador
IMAGE_SIZE = 224  # Tamanho da imagem para a rede (224x224 é padrão para muitas CNNs pré-treinadas)
NUM_WORKERS = 6  # Número de threads para carregamento paralelo de dados
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define GPU como dispositivo de processamento
TARGET_IMAGES_PER_CLASS = 300  # Número desejado de imagens por classe após augmentation

# Diretórios para salvar resultados - agora usando Path
MODELS_DIR = "models"       # Para modelos treinados
PLOTS_DIR = "plots"         # Para gráficos
REPORTS_DIR = "reports"     # Para relatórios e métricas
AUGMENTED_DIR = "augmented" # Para imagens aumentadas

# Caminho padrão para imagens aumentadas
DEFAULT_AUGMENTED_PATH = Path(r"C:\tcc\modelo\output\paper")


def create_output_directories(output_base_dir):
    """
    Cria os diretórios necessários para salvar os resultados do treinamento.

    Args:
        output_base_dir: Diretório base onde todos os outros diretórios serão criados

    Returns:
        Dicionário contendo os caminhos para cada diretório
    """
    # Convertendo para Path para melhor manipulação
    base_path = Path(output_base_dir)

    # Define os diretórios usando Path
    directories = {
        'base': str(base_path),
        'models': str(base_path / MODELS_DIR),
        'plots': str(base_path / PLOTS_DIR),
        'reports': str(base_path / REPORTS_DIR),
        'augmented': str(base_path / AUGMENTED_DIR)
    }

    # Cria cada diretório se não existir
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Diretório criado/verificado: {dir_path}")

    return directories

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


class SnakeCNN(nn.Module):
    """
    Modelo CNN baseado na arquitetura do artigo
    'A CNN Based Model for Venomous and Non-venomous Snake Classification'

    O modelo consiste em 3 camadas convolucionais seguidas por max-pooling,
    e usa uma taxa de dropout de 0.2 para evitar overfitting.
    """

    def __init__(self, num_classes):
        super(SnakeCNN, self).__init__()

        # Seguindo a arquitetura do artigo: 3 camadas conv com 16, 32 e 64 filtros
        # Cada uma seguida por ativação ReLU e max pooling
        self.features = nn.Sequential(
            # Primeiro bloco convolucional
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Segundo bloco convolucional
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Terceiro bloco convolucional
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calcula o tamanho de entrada para a primeira camada totalmente conectada
        # Para uma imagem de entrada 224x224, após 3 operações de max pooling
        # as dimensões espaciais se tornam 28x28
        fc_input_size = 64 * 28 * 28

        # Classificador seguindo a abordagem do artigo
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Artigo usa dropout de 0.2
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SnakeCNNDeeper(nn.Module):
    """
    Versão estendida da arquitetura do artigo com mais regularização
    para ajudar a prevenir overfitting para classificação multi-classe de espécies.
    """

    def __init__(self, num_classes):
        super(SnakeCNNDeeper, self).__init__()

        # Extrator de características - mais profundo do que o artigo original mas com estrutura similar
        self.features = nn.Sequential(
            # Primeiro bloco
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # Batch normalization adicionado para melhor estabilidade no treinamento
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Segundo bloco
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Terceiro bloco
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Quarto bloco (adicional)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Para uma imagem de entrada 224x224, após 4 operações de max pooling (stride 2)
        # as dimensões espaciais se tornam 14x14
        fc_input_size = 128 * 14 * 14

        # Classificador com regularização mais forte
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Dropout aumentado
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def verify_augmented_images(original_dir, augmented_dir, target_images_per_class, min_images_per_class,
                            force_augmentation=False):
    """
    Verifica se o diretório de imagens aumentadas já existe e contém dados suficientes.
    Se as imagens aumentadas não existirem ou forem insuficientes, realiza a aumentação.

    Args:
        original_dir: Diretório original com as imagens
        augmented_dir: Diretório onde devem estar as imagens aumentadas
        target_images_per_class: Número alvo de imagens por classe
        min_images_per_class: Número mínimo de imagens por classe
        force_augmentation: Se True, força a regeração das imagens mesmo se existirem

    Returns:
        str: Caminho para o diretório a ser usado para treinamento (original ou aumentado)
        bool: True se usará imagens aumentadas, False se usará originais
    """
    if force_augmentation:
        print(f"\n⚠️ Forçando regeração de imagens aumentadas...")
        augmented_dir = augment_and_save_images(original_dir, augmented_dir, target_images_per_class)
        return augmented_dir, True

    print(f"\n🔍 Verificando se imagens aumentadas já existem em: {augmented_dir}")

    # Verifica se o diretório existe
    if not os.path.exists(augmented_dir):
        print(f"  ↳ Diretório de aumentação não encontrado. Gerando imagens aumentadas...")
        augmented_dir = augment_and_save_images(original_dir, augmented_dir, target_images_per_class)
        return augmented_dir, True

    # Verifica as classes no diretório original
    original_classes = set()
    for item in os.listdir(original_dir):
        class_dir = os.path.join(original_dir, item)
        if os.path.isdir(class_dir):
            class_images = [f for f in os.listdir(class_dir)
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
        print(f"  ↳ Regenerando imagens aumentadas...")
        augmented_dir = augment_and_save_images(original_dir, augmented_dir, target_images_per_class)
        return augmented_dir, True

    # Verifica se cada classe tem imagens suficientes
    insufficient_classes = []
    for cls in original_classes:
        if cls in augmented_class_counts:
            if augmented_class_counts[cls] < target_images_per_class * 0.9:  # Tolerância de 90%
                insufficient_classes.append(f"{cls} ({augmented_class_counts[cls]} imagens)")

    if insufficient_classes:
        print(f"  ↳ Classes com imagens insuficientes: {', '.join(insufficient_classes)}")
        print(f"  ↳ Regenerando imagens aumentadas...")
        augmented_dir = augment_and_save_images(original_dir, augmented_dir, target_images_per_class)
        return augmented_dir, True

    # Verifica a integridade das imagens (opcional - apenas amostragem)
    print(f"  ↳ Verificando integridade de algumas imagens aumentadas...")

    # Amostragem aleatória de algumas imagens para verificar integridade
    corrupted_images = False
    for cls in random.sample(list(augmented_classes), min(5, len(augmented_classes))):
        class_dir = os.path.join(augmented_dir, cls)
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            continue

        # Verifica até 3 imagens aleatórias desta classe
        for img_file in random.sample(image_files, min(3, len(image_files))):
            img_path = os.path.join(class_dir, img_file)
            try:
                with Image.open(img_path) as img:
                    # Apenas carrega para verificar se não está corrompida
                    img.verify()
            except Exception as e:
                print(f"  ⚠️ Imagem corrompida encontrada: {img_path}")
                print(f"  ⚠️ Erro: {e}")
                corrupted_images = True
                break

        if corrupted_images:
            break

    if corrupted_images:
        print(f"  ⚠️ Imagens corrompidas encontradas. Regenerando imagens aumentadas...")
        augmented_dir = augment_and_save_images(original_dir, augmented_dir, target_images_per_class)
        return augmented_dir, True

    # Se chegou até aqui, o diretório está OK
    print(f"  ✅ Diretório de imagens aumentadas está completo e íntegro!")
    print(f"  ↳ Total de {len(augmented_classes)} classes com aproximadamente {target_images_per_class} imagens cada")

    # Exibe estatísticas resumidas
    total_images = sum(augmented_class_counts.values())
    avg_images = total_images / len(augmented_class_counts) if augmented_class_counts else 0

    print(f"  📊 Total de imagens aumentadas: {total_images}")
    print(f"  📊 Média de imagens por classe: {avg_images:.1f}")
    print(f"  📊 Usando imagens aumentadas existentes para treinamento...")

    # Cria um cache de verificação para registrar que as imagens foram verificadas
    verification_cache = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'original_dir': original_dir,
        'augmented_dir': augmented_dir,
        'total_classes': len(augmented_classes),
        'total_images': total_images,
        'avg_images_per_class': avg_images,
        'class_counts': augmented_class_counts
    }

    # Salva o cache de verificação no diretório aumentado
    cache_path = os.path.join(augmented_dir, 'verification_cache.json')
    try:
        with open(cache_path, 'w') as f:
            json.dump(verification_cache, f, indent=4)
    except Exception as e:
        print(f"  ⚠️ Aviso: Não foi possível salvar o cache de verificação: {e}")

    return augmented_dir, True

# Integração com o código existente
def get_snake_cnn_model(num_classes, model_variant='paper'):
    """
    Cria e retorna o modelo de classificação de serpentes.

    Args:
        num_classes: Número de espécies de serpentes para classificar
        model_variant: 'paper' para a arquitetura exata do artigo ou 'deeper' para uma versão aprimorada

    Returns:
        Modelo PyTorch
    """
    if model_variant == 'paper':
        return SnakeCNN(num_classes)
    elif model_variant == 'deeper':
        return SnakeCNNDeeper(num_classes)
    else:
        raise ValueError(f"Variante de modelo desconhecida: {model_variant}")


# Aumentação de dados especificamente projetada para imagens de serpentes
def get_snake_data_transforms():
    """
    Retorna transformações de dados especificamente projetadas para imagens de serpentes,
    com aumentação mais forte para ajudar a prevenir overfitting.
    """
    train_transform = transforms.Compose([
        # 1. Primeiro as transformações que trabalham com imagens PIL
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # Serpentes podem estar em qualquer orientação
        transforms.RandomRotation(30),
        # Color jitter com valores mais altos - serpentes podem aparecer em diferentes condições de iluminação
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        # Perspectiva aleatória e transformações afins para simular diferentes ângulos de câmera
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),

        # 2. Converter para tensor - IMPORTANTE: deve ser antes de RandomErasing e Normalize
        transforms.ToTensor(),

        # 3. Transformações que trabalham com tensores
        # Random erasing para simular oclusão parcial (ex.: quando a serpente está parcialmente escondida)
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

# Função de treinamento modificada com melhores práticas para prevenir overfitting
def train_snake_cnn(model, train_loader, val_loader, num_epochs=None, learning_rate=None,
                    weight_decay=1e-4, output_dirs=None):
    """
    Função de treinamento com práticas aprimoradas para prevenir overfitting:
    - Regularização por weight decay
    - Agendamento de taxa de aprendizado
    - Early stopping
    - Treinamento com precisão mista
    """
    # Usar valores globais se não forem fornecidos
    if num_epochs is None:
        num_epochs = NUM_EPOCHS
    if learning_rate is None:
        learning_rate = LEARNING_RATE

    if output_dirs is None:
        output_dirs = {
            'models': MODELS_DIR,
            'plots': PLOTS_DIR,
            'reports': REPORTS_DIR
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # CrossEntropyLoss para classificação multi-classe
    criterion = nn.CrossEntropyLoss()

    # Otimizador SGD com momentum e weight decay - seguindo o artigo
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    # Agendador de taxa de aprendizado para reduzir LR quando a perda de validação atinge um platô
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Corrigido: Usando a nova sintaxe para GradScaler
    scaler = torch.amp.GradScaler('cuda')

    # Histórico de treinamento
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    # Variáveis para early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0

    # Caminho para salvar o melhor modelo durante o treinamento
    checkpoint_path = os.path.join(output_dirs['models'], 'best_model_checkpoint.pth')

    # Registra o tempo de início
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n🚀 Iniciando treinamento: {timestamp}")
    print(f"📊 Monitorando progresso... Paciência para early stopping: {patience} épocas")
    print(f"⚙️ Configuração: {num_epochs} épocas, LR inicial: {learning_rate}, Weight decay: {weight_decay}")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Fase de treinamento
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zera os gradientes
            optimizer.zero_grad()

            # Treinamento com precisão mista
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward e otimização
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Estatísticas
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

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        # Atualiza o agendador
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Atualiza o histórico
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # Calcula o tempo de época
        epoch_time = time.time() - epoch_start

        # Imprime o progresso
        print(f'Época {epoch + 1}/{num_epochs} | '
              f'Tempo: {epoch_time:.1f}s | '
              f'LR: {current_lr:.6f} | '
              f'Treino: Perda {train_loss:.4f}, Acc {train_acc:.2f}% | '
              f'Val: Perda {val_loss:.4f}, Acc {val_acc:.2f}%')

        # Salva o melhor modelo (baseado na perda de validação)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0

            # Salva checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': history,
            }, checkpoint_path)

            print(f"✅ Novo melhor modelo! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"⏳ Aguardando melhoria... ({patience_counter}/{patience})")

        # Early stopping
        if patience_counter >= patience:
            print(f'🛑 Early stopping acionado após {epoch + 1} épocas')
            break

    # Cálculo do tempo total
    total_time = time.time() - start_time

    print(f"\n✅ Treinamento concluído em {total_time / 60:.2f} minutos!")
    print(f"Melhor perda de validação: {best_val_loss:.4f}")

    # Carrega os pesos do melhor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Modelo restaurado para o melhor checkpoint")

    return model, history

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

    Args:
        data_dir: Diretório original contendo as imagens
        output_dir: Diretório onde salvar as imagens aumentadas
        target_images_per_class: Número desejado de imagens por classe após augmentation

    Returns:
        Caminho para o diretório de saída com as imagens aumentadas
    """
    print("\n" + "=" * 80)
    print("🔄 AUMENTAÇÃO DE DADOS - GERANDO IMAGENS SINTÉTICAS 🔄".center(80))
    print("=" * 80)

    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Transformações para data augmentation
    augmentation_transforms = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
        transforms.GaussianBlur(kernel_size=3),
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

                # Aplicar transformações aleatórias
                random.shuffle(augmentation_transforms)
                num_transforms = random.randint(2, len(augmentation_transforms))

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


def plot_learning_curves(model_name, history, output_dirs):
    """
    Plota as curvas de aprendizado (acurácia e perda) para treinamento e validação
    para analisar possíveis sinais de overfitting.

    Args:
        model_name: Nome do modelo para o título e nome do arquivo
        history: Dicionário contendo os históricos de treino e validação
        output_dirs: Dicionário com diretórios de saída
    """
    try:
        # Extrai os dados
        train_losses = history['train_loss']
        val_losses = history['val_loss']
        train_accs = history['train_acc']
        val_accs = history['val_acc']
        epochs = range(1, len(train_losses) + 1)

        # Timestamp para o nome do arquivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
        plots_dir = output_dirs['plots']
        file_path = os.path.join(plots_dir, f'learning_curves_{model_name}_{timestamp}.png')
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        print(f"✅ Curvas de aprendizado salvas em '{file_path}'")

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

        # Salvar análise em um arquivo
        analysis_data = {
            'model_name': model_name,
            'timestamp': timestamp,
            'avg_gap': avg_gap,
            'peak_val_acc': peak_val_acc,
            'peak_epoch': peak_epoch,
            'epochs_since_improvement': epochs_since_improvement,
            'has_overfitting': avg_gap > 15,
            'recommendation': 'Aumentar regularização' if avg_gap > 7 else 'Modelo está generalizando bem'
        }

        reports_dir = output_dirs['reports']
        analysis_file = os.path.join(reports_dir, f'overfitting_analysis_{model_name}_{timestamp}.json')
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=4)

        print(f"✅ Análise de overfitting salva em '{analysis_file}'")

        return analysis_data

    except Exception as e:
        print(f"❌ Erro ao plotar curvas de aprendizado: {e}")
        return None


def generate_confusion_matrix(model, val_loader, class_names, model_name, output_dirs):
    """
    Gera uma matriz de confusão para avaliar o desempenho do modelo por classe.

    Args:
        model: Modelo treinado
        val_loader: DataLoader com dados de validação
        class_names: Lista com os nomes das classes
        model_name: Nome do modelo para salvar o arquivo
        output_dirs: Dicionário com diretórios de saída
    """
    try:
        # Coleta previsões e rótulos verdadeiros
        y_true = []
        y_pred = []

        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

        # Timestamp para o nome do arquivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

        # Salva a matriz de confusão
        plots_dir = output_dirs['plots']
        confusion_path = os.path.join(plots_dir, f'confusion_matrix_{model_name}_{timestamp}.png')
        plt.savefig(confusion_path)
        plt.close()

        print(f"✅ Matriz de confusão salva em '{confusion_path}'")

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
        reports_dir = output_dirs['reports']
        report_path = os.path.join(reports_dir, f'classification_report_{model_name}_{timestamp}.csv')
        report_df.to_csv(report_path)
        print(f"✅ Relatório de classificação salvo em '{report_path}'")

        # Salva os dados brutos da matriz de confusão
        cm_path = os.path.join(reports_dir, f'confusion_matrix_data_{model_name}_{timestamp}.csv')
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(cm_path)
        print(f"✅ Dados da matriz de confusão salvos em '{cm_path}'")

        # Salva análise por classe em JSON
        class_analysis = {
            'model_name': model_name,
            'timestamp': timestamp,
            'worst_classes': [(cls, float(acc)) for cls, acc in worst_classes],
            'best_classes': [(cls, float(acc)) for cls, acc in best_classes[::-1]],
            'accuracy_per_class': {cls: float(accuracy_per_class[i]) for i, cls in enumerate(class_names)}
        }

        class_analysis_path = os.path.join(reports_dir, f'class_analysis_{model_name}_{timestamp}.json')
        with open(class_analysis_path, 'w') as f:
            json.dump(class_analysis, f, indent=4)

        print(f"✅ Análise por classe salva em '{class_analysis_path}'")

        return {
            'worst_classes': worst_classes,
            'best_classes': best_classes,
            'report': report,
            'confusion_matrix': cm.tolist(),
            'paths': {
                'confusion_matrix': confusion_path,
                'classification_report': report_path,
                'class_analysis': class_analysis_path
            }
        }

    except Exception as e:
        print(f"❌ Erro ao gerar matriz de confusão: {e}")
        import traceback
        traceback.print_exc()
        return None


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


def parse_arguments():
    """
    Processa argumentos da linha de comando.
    """
    import argparse

    # Acessa a variável global NUM_EPOCHS
    global NUM_EPOCHS

    parser = argparse.ArgumentParser(description='Treinamento de classificador de serpentes baseado no artigo')
    parser.add_argument('--data-dir', type=str, default=r"C:\tcc\imgs-inaturalist",
                        help='Diretório contendo as imagens organizadas por classe')
    parser.add_argument('--output-dir', type=str, default=r"output",
                        help='Diretório base para salvar modelos, gráficos e relatórios')
    parser.add_argument('--augmented-dir', type=str, default=str(DEFAULT_AUGMENTED_PATH),
                        help='Diretório fixo para imagens aumentadas (sem timestamp)')
    parser.add_argument('--target-per-class', type=int, default=300,
                        help='Número alvo de imagens por classe após a aumentação')
    parser.add_argument('--min-images', type=int, default=10,
                        help='Número mínimo de imagens por classe para incluir no treinamento')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Tamanho do batch para treinamento')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,  # Usa o valor global NUM_EPOCHS
                        help=f'Número de épocas de treinamento (padrão: {NUM_EPOCHS})')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                        help='Taxa de aprendizado para o otimizador')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                        help='Número de workers para carregamento de dados')
    parser.add_argument('--skip-augmentation', action='store_true',
                        help='Pular a etapa de aumentação de dados')
    parser.add_argument('--model-variant', type=str, default='paper',
                        choices=['paper', 'deeper'],
                        help='Variante do modelo (paper: exatamente como no artigo, deeper: versão melhorada)')
    parser.add_argument('--force-augmentation', action='store_true',
                        help='Forçar a regeneração das imagens aumentadas mesmo se o diretório já existir')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='Fator de regularização L2 (weight decay)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Nome personalizado para esta execução (usado nos nomes dos arquivos)')

    args = parser.parse_args()

    # Certifica-se de que o diretório de aumentação seja um caminho absoluto
    args.augmented_dir = str(Path(args.augmented_dir).absolute())
    print(f"Diretório de aumentação configurado: {args.augmented_dir}")

    # Imprime informações de configuração para verificação
    print(f"Épocas configuradas: {args.epochs}")
    print(f"Batch size configurado: {args.batch_size}")
    print(f"Learning rate configurado: {args.learning_rate}")

    return args


def main():
    """
    Função principal que orquestra todo o processo de treinamento.
    """
    # Importações necessárias dentro da função
    import json

    # Processa argumentos da linha de comando
    args = parse_arguments()

    # Atualiza configurações globais com base nos argumentos
    global BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, NUM_WORKERS
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    NUM_WORKERS = args.workers

    # Define nome da execução
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name if args.run_name else f"{args.model_variant}_{timestamp}"

    print("\n" + "=" * 80)
    print(f"🐍 INICIANDO TREINAMENTO DO CLASSIFICADOR DE SERPENTES CNN 🐍".center(80))
    print(f"🏷️  Nome da execução: {run_name}".center(80))
    print("=" * 80 + "\n")

    # Cria e configura diretórios de saída para esta execução específica
    output_base_dir = os.path.join(args.output_dir, run_name)
    output_dirs = create_output_directories(output_base_dir)

    # IMPORTANTE: Usa um diretório fixo para as imagens aumentadas
    # Este diretório é independente do timestamp da execução
    augmented_dir = args.augmented_dir

    # Garante que o diretório existe
    os.makedirs(augmented_dir, exist_ok=True)

    print(f"📁 Diretório fixo para imagens aumentadas: {augmented_dir}")

    # Salva configuração da execução
    config = {
        'run_name': run_name,
        'timestamp': timestamp,
        'model_variant': args.model_variant,
        'data_dir': args.data_dir,
        'augmented_dir': augmented_dir,  # Salva o caminho para referência
        'target_per_class': args.target_per_class,
        'min_images_per_class': args.min_images,
        'batch_size': BATCH_SIZE,
        'epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': args.weight_decay,
        'workers': NUM_WORKERS,
        'skip_augmentation': args.skip_augmentation,
        'output_dirs': {k: v for k, v in output_dirs.items()}
    }

    config_path = os.path.join(output_dirs['base'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"💾 Configuração salva em '{config_path}'")

    # Verifica configuração da GPU
    check_cuda_status()

    # Diretório das imagens originais e aumentadas
    data_dir = args.data_dir
    target_per_class = args.target_per_class
    min_images_per_class = args.min_images

    # Etapa de aumentação de dados (opcional)
    if not args.skip_augmentation:
        try:
            # Usa a função para verificar imagens aumentadas no diretório fixo
            using_augmented = False
            if os.path.exists(augmented_dir):
                print(f"\n🔍 Verificando diretório fixo de imagens aumentadas: {augmented_dir}")

                # Verifica se é um diretório vazio (primeira execução)
                if len(os.listdir(augmented_dir)) == 0:
                    print(f"  ↳ Diretório vazio. Gerando imagens aumentadas...")
                    augment_and_save_images(data_dir, augmented_dir, target_per_class)
                    data_dir = augmented_dir
                    using_augmented = True
                # Verifica se tem o número suficiente de imagens
                elif args.force_augmentation:
                    print(f"  ↳ Forçando regeração de imagens aumentadas...")
                    augment_and_save_images(data_dir, augmented_dir, target_per_class)
                    data_dir = augmented_dir
                    using_augmented = True
                else:
                    # Verificar se as classes estão presentes e com imagens suficientes
                    valid_augmented = check_augmented_directory(data_dir, augmented_dir,
                                                                target_per_class, min_images_per_class)
                    if valid_augmented:
                        print(f"  ✅ Usando diretório de imagens aumentadas existente: {augmented_dir}")
                        data_dir = augmented_dir
                        using_augmented = True
                    else:
                        print(f"  ↳ Regenerando imagens aumentadas...")
                        augment_and_save_images(data_dir, augmented_dir, target_per_class)
                        data_dir = augmented_dir
                        using_augmented = True
            else:
                print(f"  ↳ Diretório de aumentação não encontrado. Gerando imagens aumentadas...")
                os.makedirs(augmented_dir, exist_ok=True)
                augment_and_save_images(data_dir, augmented_dir, target_per_class)
                data_dir = augmented_dir
                using_augmented = True

            # Registra o caminho final usado para treinamento
            print(f"\n📁 Diretório de dados para treinamento: {data_dir}")

        except Exception as e:
            print(f"\n⚠️ Aviso: Erro durante a etapa de aumentação de dados: {e}")
            import traceback
            traceback.print_exc()
            print(f"Continuando com as imagens originais em: {data_dir}")
            using_augmented = False
    else:
        print(f"\n⚠️ Etapa de aumentação de dados pulada. Usando diretório original: {data_dir}")
        using_augmented = False

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

    # Salva a divisão dos dados para reprodutibilidade
    dataset_split = {
        'train_count': len(train_paths),
        'val_count': len(val_paths),
        'class_distribution': {class_names[i]: train_labels.count(i) for i in range(len(class_names))},
        'class_to_idx': class_to_idx
    }

    split_path = os.path.join(output_dirs['reports'], 'dataset_split.json')
    with open(split_path, 'w') as f:
        json.dump(dataset_split, f, indent=4)

    print(f"💾 Informações de divisão do dataset salvas em '{split_path}'")

    # Obtém transformações para os dados
    print("\n🔄 Configurando transformações de imagem e aumentação de dados...")
    train_transform, val_transform = get_snake_data_transforms()

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

    # Inicializa o modelo
    num_classes = len(class_names)
    print(f"\n🧠 Inicializando modelo CNN para classificar {num_classes} espécies de serpentes...")
    print(
        f"Usando variante do modelo: {args.model_variant} - {'Seguindo exatamente o artigo' if args.model_variant == 'paper' else 'Versão melhorada'}")

    model = get_snake_cnn_model(num_classes, model_variant=args.model_variant).to(DEVICE)

    # Imprime resumo do modelo
    print("\nArquitetura do modelo:")
    print(model)

    # Salva a arquitetura do modelo em um arquivo de texto
    model_summary = []
    model_summary.append(f"Modelo: {args.model_variant}")
    model_summary.append(f"Número de classes: {num_classes}")
    model_summary.append(f"Arquitetura:\n{model}")

    # Conta parâmetros do modelo
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_summary.append(f"\nTotal de parâmetros: {total_params:,}")
    model_summary.append(f"Parâmetros treináveis: {trainable_params:,}")

    model_summary_path = os.path.join(output_dirs['models'], f'model_architecture_{args.model_variant}.txt')
    with open(model_summary_path, 'w') as f:
        f.write('\n'.join(model_summary))

    print(f"💾 Resumo da arquitetura salvo em '{model_summary_path}'")

    # Treina o modelo
    print("\n🚀 Iniciando treinamento do modelo...")
    print(
        f"Épocas: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}, Weight Decay: {args.weight_decay}")

    start_time = time.time()
    trained_model, history = train_snake_cnn(
        model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=args.weight_decay,
        output_dirs=output_dirs
    )
    training_time = time.time() - start_time

    print(f"\n✅ Treinamento concluído em {training_time / 60:.2f} minutos!")

    # Salva o modelo final
    final_model_path = os.path.join(output_dirs['models'], f'snake_cnn_{args.model_variant}_{run_name}_final.pth')
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'history': history,
        'class_names': class_names,
        'class_to_idx': class_to_idx,
        'model_variant': args.model_variant,
        'training_time': training_time,
        'parameters': {
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': args.weight_decay
        }
    }, final_model_path)

    print(f"💾 Modelo final salvo como '{final_model_path}'")

    # Avaliar desempenho
    print("\n" + "=" * 80)
    print("🔍 AVALIAÇÃO DE DESEMPENHO DO MODELO 🔍".center(80))
    print("=" * 80)

    # Gera curvas de aprendizado
    print("\n📊 Gerando curvas de aprendizado...")
    learning_curves_analysis = plot_learning_curves(args.model_variant, history, output_dirs)

    # Gera matriz de confusão
    print("\n📊 Gerando matriz de confusão...")
    confusion_analysis = generate_confusion_matrix(trained_model, val_loader, class_names, args.model_variant,
                                                   output_dirs)

    # Resumo final
    print("\n" + "=" * 80)
    print("📋 RESUMO FINAL DO TREINAMENTO 📋".center(80))
    print("=" * 80)

    best_val_acc = max(history['val_acc'])
    final_val_acc = history['val_acc'][-1]

    print(f"Modelo: CNN {'(arquitetura do artigo)' if args.model_variant == 'paper' else '(versão melhorada)'}")
    print(f"Número de classes: {num_classes}")
    print(f"Melhor acurácia de validação: {best_val_acc:.2f}%")
    print(f"Acurácia de validação final: {final_val_acc:.2f}%")
    print(f"Tempo total de treinamento: {training_time / 60:.2f} minutos")

    # Salva o resumo final
    final_results = {
        'run_name': run_name,
        'timestamp': timestamp,
        'model_variant': args.model_variant,
        'num_classes': num_classes,
        'best_validation_accuracy': best_val_acc,
        'final_validation_accuracy': final_val_acc,
        'training_time_minutes': training_time / 60,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'output_files': {
            'model': final_model_path,
            'architecture': model_summary_path
        },
        'overfitting_analysis': learning_curves_analysis if learning_curves_analysis else None
    }

    final_results_path = os.path.join(output_dirs['reports'], f'final_results_{run_name}.json')
    with open(final_results_path, 'w') as f:
        # Para garantir serialização JSON adequada
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return super().default(obj)

        json.dump(final_results, f, indent=4, cls=CustomEncoder)

    print(f"💾 Resultados finais salvos em '{final_results_path}'")

    if learning_curves_analysis and learning_curves_analysis.get('has_overfitting', False):
        print("\n⚠️ ALERTA: Detectados sinais de overfitting!")
        print(f"Gap entre treino e validação: {learning_curves_analysis['avg_gap']:.2f}%")
        print("Recomendações:")
        print("- Aumentar a regularização (weight decay)")
        print("- Aumentar o dropout")
        print("- Coletar mais dados de treinamento")
        print("- Usar técnicas de data augmentation mais agressivas")
    else:
        print("\n✅ O modelo parece estar generalizando bem!")

    print(f"\n📁 Todos os resultados foram salvos no diretório: {output_base_dir}")
    print("✨ Treinamento concluído com sucesso! ✨")

if __name__ == "__main__":
    main()