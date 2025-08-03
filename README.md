# 🐾 Transfer Learning com TensorFlow/Keras: Cães vs Gatos

Este projeto aplica técnicas de **Transfer Learning** para treinar uma rede neural capaz de distinguir **cães e gatos**, usando o dataset oficial da Microsoft/Kaggle.

## 📂 Estrutura final do projeto

```
/
├── Transfer_Learning_DIO.ipynb      # Notebook executável no Colab
├── dataset/                         # Imagens organizadas
│   ├── train/
│   │   ├── Cat/
│   │   └── Dog/
│   └── val/
├── models/                          # Modelos salvos via ModelCheckpoint
└── README.md                        # Este arquivo de documentação
```

## 📥 Etapas realizadas no notebook

1. **Download e extração dos dados**
   - Dataset baixado diretamente via Kaggle (ou outro mirror) evitando links quebrados.
   - Pastas `Cat` e `Dog` criadas com split treino (80%) e validação (20%).

2. **Filtragem de imagens inválidas**
   - Bloco de código que verifica cada imagem com `PIL.Image.open().verify()` e descarta arquivos corrompidos (com `UnidentifiedImageError`), evitando erros durante o treinamento.

3. **Pré-processamento e Augmentação**
   - Uso de `ImageDataGenerator` com `rescale`, rotação, `zoom`, flips horizontais, etc.

4. **Construção do modelo com Transfer Learning**
   - É utilizada a **MobileNetV2** pré-treinada no ImageNet como base congelada.
   - Uma camada `GlobalAveragePooling2D` seguida de `Dense` com ativação `sigmoid`.

5. **Treinamento e callbacks**
   - O modelo é compilado com otimização `Adam` e loss `binary_crossentropy`.
   - Callback de **ModelCheckpoint** para salvar a melhor versão.
   - Callback de **EarlyStopping** para parar o treino se não houver progresso.

6. **Avaliação**
   - Plot de gráficos de acurácia e perda tanto em treino quanto validação.
   - Uso do conjunto `val` para estimar generalização.

7. **Inferência**
   - Seleção de imagem de validação com predição, exibindo o rótulo previsto (gato ou cachorro) e confiança.

## 🎯 Resultados (exemplo esperado)

| Métrica                  | Valor aproximado |
|--------------------------|------------------|
| Acurácia (treino)        | ~98 %            |
| Acurácia (validação)     | ~97‑98 %         |
| Melhor modelo salvo em   | época intermediária (via ModelCheckpoint) |

## 🔧 Instruções para rodar o notebook

1. Clone este repositório.
2. Abra o notebook `Transfer_Learning_DIO.ipynb` no Google Colab.
3. Garanta que o dataset esteja na pasta `dataset/train` e `dataset/val`.
4. Execute as células sequencialmente.
5. O modelo treinado será salvo automaticamente na pasta `models/`.
6. Visualize previsões com novas imagens, alterando o caminho nos blocos de inferência.

## 📚 Referência e inspiração

Este projeto segue a abordagem da trilha da [DIO/BairesDev](https://web.dio.me/track/bairesdev-machine-learning-training), aplicando **MobileNetV2** como backbone e técnicas responsáveis de limpeza de dados e augmentação.
