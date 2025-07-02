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

# Set matplotlib backend to non-interactive mode
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
import time
import re

# Configurações globais
HIGH_RECALL_BY_DEFAULT = False  # Prioriza segurança (detecção de peçonhentas)
DEFAULT_THRESHOLD = 0.5  # Threshold mais baixo para priorizar detecção de peçonhentas


class SnakeBinaryClassifier(nn.Module):
    """
    Modelo de classificação binária para serpentes usando transfer learning.
    """

    def __init__(self, base_model='resnet50', dropout_rate=0.5):
        super(SnakeBinaryClassifier, self).__init__()

        self.base_model_name = base_model

        # Seleção do modelo base pré-treinado
        if base_model == 'resnet50':
            base = models.resnet50(weights='IMAGENET1K_V1')
            backbone = nn.Sequential(*list(base.children())[:-2])  # Remove avg pool e FC
            feature_size = 2048  # ResNet50 produz 2048 canais de features
        elif base_model == 'efficientnet_b3':
            base = models.efficientnet_b3(weights='IMAGENET1K_V1')
            backbone = base.features  # Extrator de características
            feature_size = 1536  # EfficientNet-B3 produz 1536 canais
        elif base_model == 'efficientnet_b0':
            base = models.efficientnet_b0(weights='IMAGENET1K_V1')
            backbone = base.features
            feature_size = 1280
        elif base_model == 'densenet169':
            base = models.densenet169(weights='IMAGENET1K_V1')
            backbone = base.features
            feature_size = 1664  # DenseNet169 produz 1664 canais
        elif base_model == 'vgg16':
            base = models.vgg16(weights='IMAGENET1K_V1')
            backbone = base.features  # Extrator de características VGG
            feature_size = 512  # VGG16 produz 512 canais
        else:
            raise ValueError(f"Modelo base não suportado: {base_model}")

        self.backbone = backbone

        # Camada de pooling adaptativa para garantir dimensões consistentes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classificador simplificado
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

    def forward(self, x):
        features = self.backbone(x)
        features = self.adaptive_pool(features)
        x = self.classifier(features)
        return x


def load_model(model_path):
    """
    Carrega um modelo treinado.
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

    # Extrai thresholds ótimos se disponíveis
    optimal_thresholds = model_config.get('optimal_thresholds', {'high_recall': DEFAULT_THRESHOLD})

    # Extrai taxa de dropout se disponível
    dropout_rate = model_config.get('dropout_rate', 0.5)

    # Determina o tipo do modelo
    base_model = model_config.get('base_model', 'resnet50')

    # Inicializa o modelo
    print(f"🧠 Inicializando modelo binário {base_model} para classificação de serpentes")
    model = SnakeBinaryClassifier(base_model=base_model, dropout_rate=dropout_rate)

    # Carrega os pesos
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Define o modelo para modo de avaliação

    # Define a transformação adequada
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return model, class_names, transform, optimal_thresholds


def load_image(image_path, transform):
    """
    Carrega e pré-processa uma imagem.
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


def predict_image(model, input_tensor, class_names, threshold=DEFAULT_THRESHOLD):
    """
    Faz uma predição usando o modelo, com threshold customizado.
    """
    with torch.no_grad():
        model.eval()
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    # Extrair probabilidades específicas para cada classe
    prob_nonvenomous = probabilities[0].item()  # Probabilidade de ser não peçonhenta
    prob_venomous = probabilities[1].item()  # Probabilidade de ser peçonhenta

    # Predição com threshold customizado
    is_venomous = prob_venomous >= threshold

    predicted_class = class_names[1] if is_venomous else class_names[0]

    # Calcular nível de certeza
    if is_venomous:
        confidence = prob_venomous * 100
    else:
        confidence = prob_nonvenomous * 100

    # Determinar o nível de confiança
    if confidence > 90:
        confidence_level = "ALTA"
    elif confidence > 70:
        confidence_level = "MÉDIA"
    else:
        confidence_level = "BAIXA"

    return {
        'predicted_class': predicted_class,
        'is_venomous': is_venomous,
        'confidence': confidence,
        'confidence_level': confidence_level,
        'threshold_used': threshold,
        'probabilities': {
            'nao_peconhenta': prob_nonvenomous * 100,
            'peconhenta': prob_venomous * 100
        }
    }


def visualize_prediction(image, prediction, output_path=None, expected_class=None):
    """
    Visualiza a predição em formato gráfico.
    """
    # Definir cores baseadas na predição e confiança
    if prediction['is_venomous']:
        if prediction['confidence_level'] == "ALTA":
            title_color = 'darkred'
            bar_color = 'red'
        elif prediction['confidence_level'] == "MÉDIA":
            title_color = 'orangered'
            bar_color = 'orange'
        else:
            title_color = 'darkorange'
            bar_color = 'gold'
    else:
        if prediction['confidence_level'] == "ALTA":
            title_color = 'darkgreen'
            bar_color = 'green'
        elif prediction['confidence_level'] == "MÉDIA":
            title_color = 'forestgreen'
            bar_color = 'yellowgreen'
        else:
            title_color = 'olivedrab'
            bar_color = 'lightgreen'

    # Criar figura com layout mais informativo
    fig = plt.figure(figsize=(12, 6))

    # Definir layout: imagem à esquerda, informações e gráfico à direita
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    # Painel esquerdo: Imagem original
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(image)
    ax_img.set_title('Imagem Original', fontsize=14)
    ax_img.axis('off')

    # Painel direito: dividido em info e gráfico
    gs_right = gs[1].subgridspec(2, 1, height_ratios=[1, 2])

    # Informações da classificação
    ax_info = fig.add_subplot(gs_right[0])
    ax_info.axis('off')

    # Título principal
    if prediction['is_venomous']:
        title = "PEÇONHENTA 🚨"
        recommendation = "CUIDADO! Entre em contato com especialistas."
    else:
        title = "NÃO PEÇONHENTA ✅"
        recommendation = "Provavelmente inofensiva, mas mantenha distância segura."

    ax_info.text(0.5, 0.8, title,
                 fontsize=18, fontweight='bold', color=title_color,
                 ha='center', va='center')

    # Informações de confiança
    confidence_text = f"Confiança: {prediction['confidence']:.1f}% ({prediction['confidence_level']})"
    ax_info.text(0.5, 0.55, confidence_text,
                 fontsize=14,
                 ha='center', va='center')

    # Recomendação
    ax_info.text(0.5, 0.3, recommendation,
                 fontsize=12, color='dimgray', fontstyle='italic',
                 ha='center', va='center')

    # Adicionar informação sobre classificação correta/incorreta se disponível
    if expected_class is not None:
        is_correct = (prediction['is_venomous'] and expected_class == 'peconhentas') or \
                     (not prediction['is_venomous'] and expected_class == 'naopeconhentas')

        if is_correct:
            status_text = "✓ CLASSIFICAÇÃO CORRETA"
            status_color = "darkgreen"
        else:
            status_text = "✗ CLASSIFICAÇÃO INCORRETA"
            status_color = "darkred"

        ax_info.text(0.5, 0.1, status_text,
                     fontsize=12, color=status_color, fontweight='bold',
                     ha='center', va='center')
    else:
        # Threshold usado
        threshold_text = f"Threshold: {prediction['threshold_used']:.2f}"
        ax_info.text(0.5, 0.1, threshold_text,
                     fontsize=10, color='dimgray',
                     ha='center', va='center')

    # Gráfico de barras para probabilidades
    ax_bar = fig.add_subplot(gs_right[1])

    # Criar barras para ambas as probabilidades
    bar_width = 0.4
    x_pos = [0.3, 1.0]  # Posições das barras

    # Barras
    values = [prediction['probabilities']['nao_peconhenta'],
              prediction['probabilities']['peconhenta']]

    colors = ['green', 'red']
    if prediction['is_venomous']:
        colors[1] = bar_color  # Destaque para peçonhenta
    else:
        colors[0] = bar_color  # Destaque para não peçonhenta

    bars = ax_bar.bar(x_pos, values, width=bar_width, color=colors, alpha=0.7)

    # Adicionar textos nas barras
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2., height + 2,
                    f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=12)

    # Configurar eixos
    ax_bar.set_ylim(0, 105)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(['Não Peçonhenta', 'Peçonhenta'], fontsize=12)
    ax_bar.set_ylabel('Probabilidade (%)', fontsize=12)
    ax_bar.set_title('Distribuição de Probabilidades', fontsize=14)
    ax_bar.grid(axis='y', alpha=0.3)

    # Adicionar linha para o threshold
    ax_bar.axhline(y=prediction['threshold_used'] * 100, color='purple',
                   linestyle='--', alpha=0.7,
                   label=f'Threshold ({prediction["threshold_used"] * 100:.0f}%)')
    ax_bar.legend()

    plt.tight_layout()

    # Salva a imagem se especificado
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"✅ Visualização salva em {output_path}")

    plt.close()


def get_expected_class_from_filename(filename):
    """
    Determina a classe esperada com base no nome do arquivo.

    Regras:
    - Se o nome começa com "np-", é não peçonhenta
    - Se o nome começa com "p-", é peçonhenta
    - Caso contrário, retorna None (não é possível determinar)
    """
    filename = os.path.basename(filename).lower()

    if filename.startswith("np-"):
        return "naopeconhentas"
    elif filename.startswith("p-"):
        return "peconhentas"
    else:
        return None


def process_image(image_path, model, class_names, transform, threshold, output_dir=None):
    """
    Processa uma imagem usando o modelo.
    """
    # Carrega a imagem
    original_image, input_tensor = load_image(image_path, transform)

    if original_image is None:
        return None

    # Faz a predição
    start_time = time.time()
    prediction = predict_image(model, input_tensor, class_names, threshold)
    inference_time = (time.time() - start_time) * 1000  # ms

    # Verifica a classe esperada com base no nome do arquivo
    expected_class = get_expected_class_from_filename(image_path)

    # Determina se a predição está correta (se possível)
    is_correct = None
    if expected_class is not None:
        is_correct = (prediction['is_venomous'] and expected_class == 'peconhentas') or \
                     (not prediction['is_venomous'] and expected_class == 'naopeconhentas')

    # Exibe os resultados
    image_name = os.path.basename(image_path)

    # Define cores para o terminal
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'

    # Tentar usar cores, mas ignorar erros (por exemplo, no Windows sem suporte)
    try:
        if prediction['is_venomous']:
            status = f"{RED}{BOLD}PEÇONHENTA 🚨{RESET}"
        else:
            status = f"{GREEN}{BOLD}NÃO PEÇONHENTA ✅{RESET}"
    except:
        # Fallback sem cores
        if prediction['is_venomous']:
            status = "PEÇONHENTA 🚨"
        else:
            status = "NÃO PEÇONHENTA ✅"

    # Espaçamento para formatação
    print("\n" + "=" * 60)
    print(f"Resultados para {image_name}:")
    print(f"Classificação: {status}")

    # Adiciona informação de acerto/erro se disponível
    if is_correct is not None:
        try:
            if is_correct:
                accuracy_status = f"{GREEN}✓ CORRETA{RESET}"
            else:
                accuracy_status = f"{RED}✗ INCORRETA{RESET}"
        except:
            accuracy_status = "✓ CORRETA" if is_correct else "✗ INCORRETA"

        print(f"Resultado: {accuracy_status} (esperado: {expected_class})")

    try:
        if prediction['confidence_level'] == "ALTA":
            confidence_display = f"{GREEN}{prediction['confidence']:.2f}% (ALTA){RESET}"
        elif prediction['confidence_level'] == "MÉDIA":
            confidence_display = f"{YELLOW}{prediction['confidence']:.2f}% (MÉDIA){RESET}"
        else:
            confidence_display = f"{RED}{prediction['confidence']:.2f}% (BAIXA){RESET}"
    except:
        confidence_display = f"{prediction['confidence']:.2f}% ({prediction['confidence_level']})"

    print(f"Confiança: {confidence_display}")
    print(f"Probabilidade Não Peçonhenta: {prediction['probabilities']['nao_peconhenta']:.2f}%")
    print(f"Probabilidade Peçonhenta: {prediction['probabilities']['peconhenta']:.2f}%")
    print(f"Threshold utilizado: {prediction['threshold_used']}")
    print(f"Tempo de inferência: {inference_time:.2f} ms")

    # Dica importante para segurança com baixa confiança
    if prediction['confidence'] < 70:
        print(f"\n⚠️ ATENÇÃO: Classificação com baixa confiança! Por segurança, trate como potencialmente peçonhenta.")

    # Salva a visualização
    if output_dir:
        img_name = os.path.splitext(image_name)[0]
        save_path = os.path.join(output_dir, f"{img_name}_prediction.png")
        visualize_prediction(original_image, prediction, save_path, expected_class)

    # Adiciona a informação de classe esperada e acerto à predição
    prediction['expected_class'] = expected_class
    prediction['is_correct'] = is_correct
    prediction['image_name'] = image_name

    return prediction


def process_directory(images_dir, model, class_names, transform, threshold, output_dir=None):
    """
    Processa todas as imagens em um diretório.
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
        csv_path = os.path.join(output_dir, "snake_predictions.csv")
        with open(csv_path, 'w') as f:
            f.write(
                "image,predicted_class,expected_class,is_correct,is_venomous,confidence,confidence_level,threshold_used,prob_non_venomous,prob_venomous\n")

    # Processa cada imagem
    results = {}

    # Contadores para estatísticas
    venomous_count = 0
    non_venomous_count = 0
    high_confidence_count = 0
    medium_confidence_count = 0
    low_confidence_count = 0

    # Contadores para análise de acertos/erros
    correct_count = 0
    incorrect_count = 0
    unknown_count = 0  # Para imagens sem prefixo conhecido

    # Contadores por categoria
    true_positives = 0  # Previsto peçonhenta, realmente peçonhenta
    false_positives = 0  # Previsto peçonhenta, realmente não peçonhenta
    true_negatives = 0  # Previsto não peçonhenta, realmente não peçonhenta
    false_negatives = 0  # Previsto não peçonhenta, realmente peçonhenta

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        print(f"[{i + 1}/{len(image_files)}] Processando {image_file}...")

        prediction = process_image(
            image_path, model, class_names, transform, threshold, output_dir
        )

        if prediction:
            results[image_file] = prediction

            # Atualiza contadores
            if prediction['is_venomous']:
                venomous_count += 1
            else:
                non_venomous_count += 1

            # Contagem por nível de confiança
            if prediction['confidence_level'] == "ALTA":
                high_confidence_count += 1
            elif prediction['confidence_level'] == "MÉDIA":
                medium_confidence_count += 1
            else:
                low_confidence_count += 1

            # Atualiza contadores de acertos/erros
            if prediction['is_correct'] is True:
                correct_count += 1
                # Atualiza contadores por categoria
                if prediction['is_venomous']:
                    true_positives += 1  # Corretamente identificado como peçonhenta
                else:
                    true_negatives += 1  # Corretamente identificado como não peçonhenta
            elif prediction['is_correct'] is False:
                incorrect_count += 1
                # Atualiza contadores por categoria
                if prediction['is_venomous']:
                    false_positives += 1  # Incorretamente identificado como peçonhenta
                else:
                    false_negatives += 1  # Incorretamente identificado como não peçonhenta
            else:
                unknown_count += 1  # Não foi possível determinar

            # Salva os resultados no CSV
            if output_dir:
                with open(csv_path, 'a') as f:
                    row = [
                        image_file,
                        prediction['predicted_class'],
                        str(prediction['expected_class']) if prediction['expected_class'] else "desconhecido",
                        str(prediction['is_correct']) if prediction['is_correct'] is not None else "desconhecido",
                        str(prediction['is_venomous']),
                        f"{prediction['confidence']:.2f}",
                        prediction['confidence_level'],
                        f"{prediction['threshold_used']:.2f}",
                        f"{prediction['probabilities']['nao_peconhenta']:.2f}",
                        f"{prediction['probabilities']['peconhenta']:.2f}"
                    ]
                    f.write(','.join(row) + '\n')

    # Exibe estatísticas finais
    total = venomous_count + non_venomous_count
    if total > 0:
        print("\n" + "=" * 60)
        print("📊 ESTATÍSTICAS DE CLASSIFICAÇÃO".center(60))
        print("=" * 60)
        print(f"Total de imagens: {total}")
        print(f"Cobras peçonhentas detectadas: {venomous_count} ({venomous_count / total * 100:.1f}%)")
        print(f"Cobras não peçonhentas detectadas: {non_venomous_count} ({non_venomous_count / total * 100:.1f}%)")

        # Adicionar estatísticas de acertos/erros
        print("\nEstatísticas de acertos e erros:")
        verificable_count = correct_count + incorrect_count
        if verificable_count > 0:
            print(f"  - Total de imagens com nome identificável: {verificable_count}")
            print(f"  - Classificações corretas: {correct_count} ({correct_count / verificable_count * 100:.1f}%)")
            print(
                f"  - Classificações incorretas: {incorrect_count} ({incorrect_count / verificable_count * 100:.1f}%)")

            # Contagem explícita da matriz de confusão
            print("\nMatriz de Confusão:")
            print(f"  - Verdadeiros Positivos (VP): {true_positives} (cobras peçonhentas corretamente identificadas)")
            print(
                f"  - Falsos Positivos (FP): {false_positives} (cobras não peçonhentas classificadas como peçonhentas)")
            print(
                f"  - Verdadeiros Negativos (VN): {true_negatives} (cobras não peçonhentas corretamente identificadas)")
            print(
                f"  - Falsos Negativos (FN): {false_negatives} (cobras peçonhentas classificadas como não peçonhentas)")

            # Métricas detalhadas
            print("\nMétricas detalhadas:")

            # Precisão: VP / (VP + FP)
            precision = true_positives / (true_positives + false_positives) if (
                                                                                           true_positives + false_positives) > 0 else 0
            print(f"  - Precisão (cobras peçonhentas): {precision:.4f}")

            # Recall/Sensibilidade: VP / (VP + FN)
            recall = true_positives / (true_positives + false_negatives) if (
                                                                                        true_positives + false_negatives) > 0 else 0
            print(f"  - Recall (cobras peçonhentas): {recall:.4f}")

            # Especificidade: VN / (VN + FP)
            specificity = true_negatives / (true_negatives + false_positives) if (
                                                                                             true_negatives + false_positives) > 0 else 0
            print(f"  - Especificidade: {specificity:.4f}")

            # F1-Score: 2 * (precisão * recall) / (precisão + recall)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print(f"  - F1-Score: {f1_score:.4f}")

        if unknown_count > 0:
            print(f"  - Imagens sem prefixo identificável: {unknown_count}")

        print("\nNíveis de confiança:")
        print(f"  - Alta confiança: {high_confidence_count} ({high_confidence_count / total * 100:.1f}%)")
        print(f"  - Média confiança: {medium_confidence_count} ({medium_confidence_count / total * 100:.1f}%)")
        print(f"  - Baixa confiança: {low_confidence_count} ({low_confidence_count / total * 100:.1f}%)")

        # Salvar um relatório completo em JSON
        if output_dir:
            report_path = os.path.join(output_dir, "analysis_report.json")
            report_data = {
                "total_images": total,
                "venomous_detected": venomous_count,
                "non_venomous_detected": non_venomous_count,
                "accuracy_metrics": {
                    "correct": correct_count,
                    "incorrect": incorrect_count,
                    "unknown": unknown_count,
                    "accuracy": correct_count / verificable_count if verificable_count > 0 else 0,
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "true_negatives": true_negatives,
                    "false_negatives": false_negatives,
                    "precision": precision,
                    "recall": recall,
                    "specificity": specificity,
                    "f1_score": f1_score
                },
                "confidence_levels": {
                    "high": high_confidence_count,
                    "medium": medium_confidence_count,
                    "low": low_confidence_count
                }
            }

            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=4)

            print(f"\n✅ Relatório detalhado salvo em: {report_path}")

    return results


def main():
    """
    Função principal para execução da inferência.
    """
    parser = argparse.ArgumentParser(description='Classificação de Serpentes (Peçonhentas/Não Peçonhentas)')
    parser.add_argument('--model', type=str, required=True, help='Caminho para o modelo treinado')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Caminho para uma imagem')
    group.add_argument('--dir', type=str, help='Caminho para um diretório contendo imagens')

    parser.add_argument('--output', type=str, help='Diretório para salvar os resultados')
    parser.add_argument('--threshold', type=float, help='Threshold personalizado para classificação (entre 0 e 1)')
    parser.add_argument('--standard', action='store_true',
                        help='Usar threshold padrão (0.5) em vez do threshold otimizado')
    parser.add_argument('--high-recall', action='store_true', help='Usar threshold otimizado para alta recall')

    args = parser.parse_args()

    # Carrega o modelo
    model, class_names, transform, optimal_thresholds = load_model(args.model)

    # Decidir qual threshold usar
    if args.threshold is not None:
        if 0 <= args.threshold <= 1:
            threshold = args.threshold
            print(f"⚙️ Usando threshold personalizado: {threshold}")
        else:
            print(f"⚠️ Threshold inválido {args.threshold}. Deve estar entre 0 e 1. Usando valores padrão.")
            threshold = DEFAULT_THRESHOLD
    elif args.standard:
        threshold = DEFAULT_THRESHOLD
        print(f"⚙️ Usando threshold padrão: {threshold}")
    elif args.high_recall:
        threshold = optimal_thresholds.get('high_recall', DEFAULT_THRESHOLD)
        print(f"⚙️ Usando threshold otimizado para segurança: {threshold}")
    elif HIGH_RECALL_BY_DEFAULT:
        # Usa o limiar otimizado por padrão para priorizar segurança
        threshold = optimal_thresholds.get('high_recall', DEFAULT_THRESHOLD)
        print(f"⚙️ Usando threshold otimizado para segurança: {threshold}")
    else:
        # Usa threshold padrão
        threshold = DEFAULT_THRESHOLD
        print(f"⚙️ Usando threshold padrão: {threshold}")

    # Cria o diretório de saída se necessário
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Processa uma imagem ou um diretório
    if args.image:
        process_image(args.image, model, class_names, transform, threshold, args.output)
    else:
        process_directory(args.dir, model, class_names, transform, threshold, args.output)


if __name__ == '__main__':
    main()