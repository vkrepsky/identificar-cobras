import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import sys
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from torchvision.models import ResNet18_Weights, VGG16_Weights, DenseNet121_Weights, Inception_V3_Weights


class SnakeClassifier(nn.Module):
    """
    Modelo de classificação de serpentes que suporta várias arquiteturas.
    Utiliza transfer learning com fine-tuning da última camada.
    """

    def __init__(self, num_classes, model_name='resnet18'):
        """
        Inicializa o modelo com a arquitetura base escolhida e adapta para a tarefa específica.

        Args:
            num_classes: Número de espécies de serpentes (classes) a classificar
            model_name: Nome da arquitetura base a ser usada
        """
        super(SnakeClassifier, self).__init__()

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
                nn.Dropout(0.5),
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
                nn.Dropout(0.5),
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
                nn.Dropout(0.5),
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
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

            # Substitui o classificador auxiliar também
            self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, num_classes)

            # Flag para rastrear se este é o modelo Inception v3 (para tratamento especial no forward)
            self.is_inception = True

        elif model_name == 'yolo_classifier':
            # YOLO adaptado para classificação (usando darknet como inspiração)
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            num_features = self.model.fc.in_features

            # Congela os parâmetros da rede base
            for param in self.model.parameters():
                param.requires_grad = False

            # Substitui o classificador com estrutura inspirada em YOLO
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 1024),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

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
        # Tratamento especial para o modelo Inception v3
        if hasattr(self, 'is_inception') and self.is_inception:
            # Durante o treinamento, Inception v3 retorna tupla (saída principal, saída auxiliar)
            if self.training:
                output, aux_output = self.model(x)
                return output  # Durante o treinamento, ignoramos a saída auxiliar
            else:
                return self.model(x)  # No modo de avaliação, apenas retorna a saída principal
        else:
            return self.model(x)


def load_model(model_path, class_info_path):
    """
    Carrega o modelo treinado e as informações das classes.

    Args:
        model_path: Caminho para o arquivo do modelo (.pth)
        class_info_path: Caminho para o arquivo de informações das classes (.json)

    Returns:
        model: Modelo carregado
        class_names: Lista com os nomes das classes
        transform: Transformações para pré-processamento
    """
    print(f"📂 Carregando informações das classes de {class_info_path}")
    with open(class_info_path, 'r') as f:
        class_info = json.load(f)

    class_names = class_info["class_names"]
    model_name = class_info["model_name"]
    num_classes = len(class_names)

    print(f"🧠 Inicializando modelo {model_name} para {num_classes} classes")
    # Criar o modelo com a arquitetura correta
    model = SnakeClassifier(num_classes, model_name=model_name)

    print(f"📂 Carregando pesos do modelo de {model_path}")
    # Carregar os pesos do modelo treinado
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Definir o modelo para modo de avaliação
    model.eval()

    # Obter as transformações corretas para o modelo
    if model_name == 'inception_v3':
        image_size = 299  # Inception v3 espera 299x299
    else:
        image_size = 224  # 224x224 para outros modelos

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return model, class_names, transform


def load_image(image_path, transform=None):
    """
    Carrega uma imagem de um caminho local ou URL e aplica transformações.

    Args:
        image_path: Caminho local ou URL da imagem
        transform: Transformações a serem aplicadas

    Returns:
        original_image: Imagem original para visualização
        input_tensor: Tensor pré-processado para o modelo
    """
    if image_path.startswith('http'):
        # Carregar imagem da internet
        print(f"🌐 Baixando imagem de {image_path}")
        response = requests.get(image_path)
        original_image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        # Carregar imagem local
        print(f"📂 Carregando imagem de {image_path}")
        original_image = Image.open(image_path).convert('RGB')

    # Aplicar transformações
    if transform:
        input_tensor = transform(original_image)
        # Adicionar dimensão de batch
        input_tensor = input_tensor.unsqueeze(0)
    else:
        input_tensor = None

    return original_image, input_tensor


def predict_image(model, input_tensor, class_names, topk=3):
    """
    Faz a predição da classe da imagem usando o modelo.

    Args:
        model: Modelo treinado
        input_tensor: Tensor de entrada
        class_names: Lista com os nomes das classes
        topk: Número de classes principais a retornar

    Returns:
        topk_probs: Probabilidades das principais classes
        topk_classes: Nomes das principais classes
    """
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    # Obter as top-k probabilidades e índices
    topk_probs, topk_indices = torch.topk(probabilities, k=min(topk, len(class_names)))

    # Converter para numpy
    topk_probs = topk_probs.cpu().numpy()
    topk_indices = topk_indices.cpu().numpy()

    # Mapear índices para nomes de classes
    topk_classes = [class_names[idx] for idx in topk_indices]

    return topk_probs, topk_classes


def visualize_prediction(image, topk_probs, topk_classes):
    """
    Visualiza a imagem com as previsões do modelo.

    Args:
        image: Imagem original
        topk_probs: Probabilidades das principais classes
        topk_classes: Nomes das principais classes

    Returns:
        fig: Objeto da figura para salvar ou exibir
    """
    # Configurar a figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Mostrar a imagem
    ax1.imshow(image)
    ax1.set_title('Imagem')
    ax1.axis('off')

    # Mostrar o gráfico de barras das probabilidades
    y_pos = np.arange(len(topk_classes))
    bars = ax2.barh(y_pos, topk_probs * 100, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(topk_classes)
    ax2.set_xlabel('Probabilidade (%)')
    ax2.set_title('Top Predições')

    # Colorir as barras baseado na confiança
    for i, bar in enumerate(bars):
        if topk_probs[i] > 0.8:  # Alta confiança (> 80%)
            bar.set_color('green')
        elif topk_probs[i] > 0.5:  # Média confiança (50-80%)
            bar.set_color('orange')
        else:  # Baixa confiança (< 50%)
            bar.set_color('red')

    # Adicionar valores percentuais nas barras
    for i, v in enumerate(topk_probs):
        ax2.text(v * 100 + 1, i, f"{v * 100:.1f}%", va='center')

    plt.tight_layout()
    return fig


def get_image_files_from_directory(directory):
    """
    Obtém uma lista de todos os arquivos de imagem em um diretório.

    Args:
        directory: Caminho para o diretório

    Returns:
        Lista de caminhos completos para arquivos de imagem
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_files = []

    # Verificar se o diretório existe
    if not os.path.isdir(directory):
        print(f"❌ Erro: O diretório {directory} não existe!")
        return []

    # Listar todos os arquivos com extensões de imagem válidas
    for filename in os.listdir(directory):
        if filename.lower().endswith(valid_extensions):
            image_files.append(os.path.join(directory, filename))

    print(f"📁 Encontradas {len(image_files)} imagens no diretório {directory}")
    return sorted(image_files)  # Ordenar arquivos para processamento consistente


def main():
    """
    Função principal para testar uma imagem ou um diretório de imagens com o modelo.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Testar o classificador de serpentes com novas imagens')
    parser.add_argument('--model', type=str, required=True, help='Caminho para o arquivo do modelo (.pth)')
    parser.add_argument('--class_info', type=str, required=True,
                        help='Caminho para o arquivo de informações das classes (.json)')
    parser.add_argument('--image', type=str, help='Caminho ou URL da imagem para testar')
    parser.add_argument('--dir', type=str, help='Diretório contendo imagens para testar')
    parser.add_argument('--topk', type=int, default=3, help='Número de classes principais a mostrar')
    parser.add_argument('--save', action='store_true', help='Salvar as visualizações em vez de exibi-las')
    parser.add_argument('--output_dir', type=str, default='resultados', help='Diretório para salvar as visualizações')
    args = parser.parse_args()

    # Verificar se pelo menos uma opção de entrada foi fornecida
    if not args.image and not args.dir:
        print("❌ Erro: Você deve fornecer uma imagem (--image) ou um diretório de imagens (--dir)")
        return

    # Carregar o modelo e as informações das classes
    model, class_names, transform = load_model(args.model, args.class_info)

    # Criar diretório de saída se a opção de salvar estiver ativada
    if args.save and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"📁 Criado diretório de saída: {args.output_dir}")

    # Caso 1: Processar uma única imagem
    if args.image:
        # Carregar e pré-processar a imagem
        original_image, input_tensor = load_image(args.image, transform)

        # Fazer a predição
        print("🔄 Processando a imagem com o modelo...")
        topk_probs, topk_classes = predict_image(model, input_tensor, class_names, args.topk)

        # Mostrar resultados
        print("\n📊 Resultados da Predição:")
        for i, (cls, prob) in enumerate(zip(topk_classes, topk_probs)):
            print(f"{i + 1}. {cls}: {prob * 100:.2f}%")

        # Visualizar a predição
        fig = visualize_prediction(original_image, topk_probs, topk_classes)

        # Salvar ou mostrar a visualização
        if args.save:
            img_name = os.path.basename(args.image) if not args.image.startswith('http') else 'online_image.jpg'
            output_path = os.path.join(args.output_dir, f"resultado_{img_name}")
            fig.savefig(output_path)
            print(f"💾 Visualização salva em: {output_path}")
        else:
            plt.show()

    # Caso 2: Processar todas as imagens em um diretório
    elif args.dir:
        # Obter lista de imagens no diretório
        image_files = get_image_files_from_directory(args.dir)

        if not image_files:
            print("❌ Nenhuma imagem encontrada no diretório.")
            return

        # Criar arquivo CSV para resultados
        results_file = os.path.join(args.output_dir, 'resultados_predicao.csv') if args.save else None
        if args.save:
            with open(results_file, 'w') as f:
                f.write('imagem,classe1,prob1,classe2,prob2,classe3,prob3\n')

        # Processar cada imagem
        for i, image_path in enumerate(image_files):
            print(f"\n{'=' * 50}")
            print(f"Imagem {i + 1}/{len(image_files)}: {os.path.basename(image_path)}")
            print(f"{'=' * 50}")

            try:
                # Carregar e pré-processar a imagem
                original_image, input_tensor = load_image(image_path, transform)

                # Fazer a predição
                topk_probs, topk_classes = predict_image(model, input_tensor, class_names, args.topk)

                # Mostrar resultados
                print("\nResultados da Predição:")
                for j, (cls, prob) in enumerate(zip(topk_classes, topk_probs)):
                    print(f"{j + 1}. {cls}: {prob * 100:.2f}%")

                # Salvar resultados no CSV
                if args.save:
                    with open(results_file, 'a') as f:
                        result_row = [os.path.basename(image_path)]
                        for cls, prob in zip(topk_classes, topk_probs):
                            result_row.extend([cls, f"{prob * 100:.2f}"])
                        # Preencher valores vazios se não houver k classes
                        while len(result_row) < 7:  # Imagem + 3 classes com probas
                            result_row.extend(['', ''])
                        f.write(','.join(result_row) + '\n')

                # Visualizar a predição
                fig = visualize_prediction(original_image, topk_probs, topk_classes)

                # Salvar ou mostrar a visualização
                if args.save:
                    output_path = os.path.join(args.output_dir, f"resultado_{os.path.basename(image_path)}")
                    fig.savefig(output_path)
                    print(f"💾 Visualização salva em: {output_path}")
                    plt.close(fig)  # Fechar figura para liberar memória
                else:
                    plt.show()

            except Exception as e:
                print(f"❌ Erro ao processar a imagem {image_path}: {e}")
                continue

        if args.save:
            print(f"\n✅ Processamento concluído! Resultados salvos em:")
            print(f"  📊 CSV: {results_file}")
            print(f"  🖼️ Visualizações: {args.output_dir}")


def test_multiple_images(model_path, class_info_path, image_paths, topk=3):
    """
    Testa múltiplas imagens com o modelo.

    Args:
        model_path: Caminho para o arquivo do modelo (.pth)
        class_info_path: Caminho para o arquivo de informações das classes (.json)
        image_paths: Lista de caminhos ou URLs de imagens
        topk: Número de classes principais a mostrar
    """
    # Carregar o modelo e as informações das classes
    model, class_names, transform = load_model(model_path, class_info_path)

    # Processar cada imagem
    for i, image_path in enumerate(image_paths):
        print(f"\n{'=' * 50}")
        print(f"Imagem {i + 1}/{len(image_paths)}: {image_path}")
        print(f"{'=' * 50}")

        # Carregar e pré-processar a imagem
        original_image, input_tensor = load_image(image_path, transform)

        # Fazer a predição
        topk_probs, topk_classes = predict_image(model, input_tensor, class_names, topk)

        # Mostrar resultados
        print("\nResultados da Predição:")
        for j, (cls, prob) in enumerate(zip(topk_classes, topk_probs)):
            print(f"{j + 1}. {cls}: {prob * 100:.2f}%")

        # Visualizar a predição
        visualize_prediction(original_image, topk_probs, topk_classes)


if __name__ == "__main__":
    # Se não houver argumentos, executar o exemplo
    if len(sys.argv) == 1:
        print("⚠️ Executando modo de exemplo, sem argumentos fornecidos")
        print("Para uso correto, execute com argumentos. Exemplo:")
        print("\n# Testar uma única imagem:")
        print(
            "python inferencia.py --model best_model_resnet18.pth --class_info class_info_resnet18.json --image https://exemplo.com/imagem.jpg")

        print("\n# Testar todas as imagens em um diretório:")
        print(
            "python inferencia.py --model best_model_resnet18.pth --class_info class_info_resnet18.json --dir pasta_de_imagens")

        print("\n# Testar um diretório e salvar os resultados:")
        print(
            "python inferencia.py --model best_model_resnet18.pth --class_info class_info_resnet18.json --dir pasta_de_imagens --save --output_dir resultados")

        print("\n❌ Este é apenas um exemplo de uso. Execute o script com os parâmetros apropriados.")
    else:
        main()

