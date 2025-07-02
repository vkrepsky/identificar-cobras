import torch
import json

def extrair_thresholds_do_modelo(modelo_path):
    """
    Extrai e exibe os thresholds otimizados salvos no modelo.
    """
    print(f"📂 Carregando modelo de {modelo_path}")
    
    try:
        # Carregar o checkpoint
        checkpoint = torch.load(modelo_path, map_location=torch.device('cpu'), weights_only=True)
        
        # Buscar thresholds em diferentes locais possíveis
        optimal_thresholds = None
        
        # Primeiro: verificar em model_config
        model_config = checkpoint.get('model_config', {})
        if 'optimal_thresholds' in model_config:
            optimal_thresholds = model_config['optimal_thresholds']
            print("✅ Thresholds encontrados em model_config")
        
        # Segundo: verificar diretamente no checkpoint
        elif 'optimal_thresholds' in checkpoint:
            optimal_thresholds = checkpoint['optimal_thresholds']
            print("✅ Thresholds encontrados no checkpoint principal")
        
        if optimal_thresholds:
            print("\n" + "="*60)
            print("📊 THRESHOLDS OTIMIZADOS ENCONTRADOS")
            print("="*60)
            
            for nome, valor in optimal_thresholds.items():
                print(f"  - {nome}: {valor:.4f}")
                
                # Explicar cada threshold
                if nome == 'youden_j':
                    print(f"    (Índice de Youden - maximiza TPR + TNR)")
                elif nome == 'min_distance':
                    print(f"    (Distância mínima ao ponto ideal (0,1) na curva ROC)")
                elif nome == 'max_f1':
                    print(f"    (Maximiza F1-Score)")
                elif nome == 'high_recall':
                    print(f"    ⭐ (ESTE é o threshold para ≥95% recall - usado para segurança)")
            
            print("\n" + "="*60)
            print("🎯 RECOMENDAÇÃO PARA SEGURANÇA:")
            print(f"   Use o threshold 'high_recall': {optimal_thresholds.get('high_recall', 'N/A')}")
            print("   Este garante pelo menos 95% de detecção de serpentes peçonhentas")
            print("="*60)
            
            # Salvar em JSON para referência
            with open('thresholds_extraidos.json', 'w') as f:
                json.dump(optimal_thresholds, f, indent=4)
            print(f"\n💾 Thresholds salvos em: thresholds_extraidos.json")
            
        else:
            print("❌ Nenhum threshold otimizado encontrado no modelo")
            print("   O modelo pode ter sido salvo sem os thresholds otimizados")
        
        # Mostrar outras informações úteis
        print(f"\n📋 OUTRAS INFORMAÇÕES DO MODELO:")
        if 'class_names' in checkpoint:
            print(f"   Classes: {checkpoint['class_names']}")
        if 'val_acc' in checkpoint:
            print(f"   Acurácia de validação: {checkpoint['val_acc']:.2f}%")
        if 'fold' in checkpoint:
            print(f"   Fold: {checkpoint['fold']}")
            
    except Exception as e:
        print(f"❌ Erro ao carregar o modelo: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Uso: python verificar_threshold.py <caminho_do_modelo.pth>")
        print("\nExemplo:")
        print("python verificar_threshold.py best_fold_5_model.pth")
        sys.exit(1)
    
    modelo_path = sys.argv[1]
    extrair_thresholds_do_modelo(modelo_path)