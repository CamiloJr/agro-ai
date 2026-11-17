# AGRO IA — Laboratório Interativo de Inspeção Foliar

**Software didático em Python para a Aula 7 — Aplicações Contemporâneas da Inteligência Artificial**  
**Ambiente:** Windows 10/11 (64 bits), Python 3.14 compatível; também é possível executar em outros sistemas com dependências instaladas.

## O que é

Aplicação visual com **webcam ou imagem do computador**, composta por dois processos **distintos**:

1. **IA real: classificação de folhas** por rede neural convolucional MobileNetV3, pré-treinada no conjunto PlantVillage. O modelo retorna três hipóteses e escores *softmax* (15 classes disponíveis).
2. **Processamento convencional: mapa de possíveis alterações cromáticas** em HSV, ilustrando manchas amareladas e áreas escuras. Isso **não é um modelo treinado para segmentar doenças** e não fornece extensão ou severidade de uma doença.

O sistema inclui seleção de região de interesse (ROI), análise de brilho e nitidez, controle de sensibilidade da máscara, opacidade, limiar de confiança, webcam, exportação de PNG e relatório JSON, além de painel explicativo.

**Não há imagens de exemplo nem botões de exemplos.**

## Instalação no Windows

1. Extraia o ZIP inteiro para uma pasta com permissão de gravação (por exemplo, Downloads).
2. Execute `INICIAR_WINDOWS.bat` (salvo em ASCII, CRLF, sem BOM).
3. A primeira execução cria `.venv` e instala as dependências (`pygame-ce`, `opencv-python`, `numpy`, `Pillow`, `onnxruntime`). É necessária internet nessa etapa.
4. Ao analisar a primeira imagem, o software baixa **dois arquivos do modelo** em `modelos/`: `model.onnx` (~293 kB) e `model.onnx.data` (~6 MB). **São obrigatórios os dois arquivos na mesma pasta.** Essa operação ocorre em segundo plano e pode exigir acesso ao domínio `huggingface.co`.
5. Depois de instaladas as bibliotecas e baixados os pesos, o classificador funciona **localmente**, sem enviar imagens para servidores.

Em versões prévias do software, problemas de BOM em `.bat` causavam o erro `'´╗┐@echo'`. O iniciador deste projeto foi gerado sem BOM.

### Execução manual alternativa

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe agroia_folhas.py
```

Se o download do modelo falhar, abra estes links em um navegador e salve os dois arquivos na pasta `modelos/` (com os nomes exatos):

- https://huggingface.co/imaflower/plantvillage-mobilenetv3/resolve/main/model.onnx
- https://huggingface.co/imaflower/plantvillage-mobilenetv3/resolve/main/model.onnx.data

A análise HSV poderá funcionar mesmo que o classificador esteja indisponível. O sistema não inventa uma previsão se a rede não estiver carregada.

## Operação

| Comando | Resultado |
|---|---|
| `Abrir imagem` / **O** | Seleciona JPG, PNG, BMP, WebP ou TIFF |
| `Iniciar webcam` / **W** | Liga a webcam, processando quadros periodicamente (aprox. a cada 2 s) |
| `Capturar` / **ESPAÇO** | Congela o quadro atual e encerra a câmera |
| `Selecionar folha / ROI` / **R** | Habilita seleção de retângulo por arrastar o mouse sobre a imagem congelada |
| `Limpar ROI` / **Delete** | Restaura a área inteira |
| `Original / Sintomas / Mapa` / **1, 2, 3** | Alterna as visualizações |
| `Reanalisar` | Solicita nova inferência |
| `Sensibilidade da máscara` | Ajusta limiares HSV; **não** altera parâmetros da rede |
| `Opacidade` | Regula a sobreposição colorida |
| `Confiança mínima` | Destaca resultados incertos; **não recalibra** o classificador |
| `Salvar PNG + JSON` / **S** | Exporta imagem e relatório para `relatorios/` |
| `Ajuda` / **H** ou **F1** | Mostra o roteiro das seis etapas |

### Leitura da interface

A coluna da direita mostra: (i) top 3 do modelo treinado, (ii) contagem percentual **de pixels coloridos** na região analisada, (iii) indicadores de qualidade da foto, e (iv) sequência do processamento. O **score softmax não é probabilidade calibrada da doença estar presente** e não deve ser interpretado como diagnóstico fitossanitário.

**Limiar de confiança:** se a maior saída for inferior ao limite configurado, aparece o aviso de incerteza. Um escore alto também não garante que a categoria esteja correta, especialmente se a folha estiver fora das classes de treinamento.

### Culturas e classes (conjunto limitado)

- **Pimentão:** mancha bacteriana, saudável;
- **Batata:** pinta-preta/alternariose, requeima, saudável;
- **Tomate:** mancha bacteriana, pinta-preta/alternariose, requeima, mofo foliar, septoriose, **dano associado a ácaros**, mancha-alvo, vírus do enrolamento amarelo, vírus do mosaico, saudável.

**Atenção:** a classe de ácaros é um padrão visual nas folhas. O software **não detecta ácaros individualmente**, não identifica outras pragas nem cobre soja, milho, café ou cana. A presença de fungo/bactéria/vírus precisa de avaliação agronômica; nenhuma recomendação química ou dosagem é fornecida.

## Roteiro rápido para a aula (8–12 minutos)

1. Exiba uma foto de folha de tomate e comente as três primeiras hipóteses da CNN.
2. Alternando **Original** e **Sintomas (HSV)**, mostre o que ocorre com amarelamento e pixels escuros.
3. Alterne para **Mapa cromático** e aumente a **Sensibilidade**. Pergunte: *o algoritmo aprendeu o que é doença ou usamos regras de cor?*
4. Faça uma ROI na folha e compare as hipóteses com a imagem inteira. Isso mostra o impacto do contexto e do pré-processamento.
5. Use a webcam e explique que o modelo é aplicado a quadros independentes, **sem rastreamento temporal de sintomas**.
6. Finalize abordando generalização: plantas fora do treinamento podem receber uma classe indevida com score alto.

## Arquitetura do software

```text
AgroIA_Folhas_Aula_7/
├── INICIAR_WINDOWS.bat       # instalador + inicializador sem BOM
├── agroia_folhas.py          # interface Pygame, webcam, ROI e relatórios
├── analise_foliar.py         # heurísticas HSV e qualidade da imagem
├── modelo_foliar.py          # carregamento ONNX e classificação real
├── requirements.txt
├── modelos/                  # cache local de pesos (primeiro uso)
├── relatorios/               # relatórios produzidos pelo usuário
├── tests/                    # testes automatizados offline
└── README.md
```

## Modelos, licença e referências

- **Modelo:** `imaflower/plantvillage-mobilenetv3`, classe MobileNetV3-Small (1,53 milhão de parâmetros), exportação ONNX, **licença MIT**. Veja https://huggingface.co/imaflower/plantvillage-mobilenetv3. Os pesos *não estão distribuídos neste ZIP*: o download ocorre diretamente do repositório do autor.
- **Dataset e limitação:** PlantVillage usa imagens controladas de folhas; fotos reais com fundo, iluminação variável e doenças não contempladas podem produzir previsões incorretas.
- **Inferência:** entrada RGB transformada em 224×224 (recorte central), normalização ImageNet e tensor float32 NCHW; saída logits e top 3 softmax.
- **ONNX Runtime:** https://onnxruntime.ai/; pacote com wheels Windows Python 3.14 a partir de versões atuais.

## Limitações e privacidade

As predições devem ser tratadas como **triagem didática, não diagnóstico**. Não use o programa para recomendar aplicação de defensivos. Não há armazenamento automático das imagens: somente `Salvar PNG + JSON` grava arquivos localmente. A webcam funciona no computador do usuário. O download dos pesos acessa o repositório público do modelo, sem enviar fotografias.

## Resolução de problemas

- **Webcam indisponível:** libere a permissão do dispositivo no Windows; feche aplicativos que já usam a câmera; teste `Capturar` após iniciar.
- **Falha ao baixar arquivos:** confira internet, proxy e bloqueio do `huggingface.co`; baixe ambos manualmente pelos links acima.
- **Erro ONNX de dados externos:** verifique os nomes `model.onnx` e `model.onnx.data` dentro da mesma pasta `modelos/`.
- **Python 3.14:** não é necessário instalar Python 3.12/3.13; execute com `py -3.14`. Se uma `.venv` antiga estiver incompatível, renomeie-a e rode o `.bat` novamente.
- **Dependências:** no terminal dentro da pasta, execute `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- **Erros internos:** verifique `agroia_erros.log` na pasta da aplicação e envie o texto completo para depuração.

### Testes

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

**Validação disponível na entrega:** testes automatizados de algoritmo e estrutura, sem necessidade de webcam ou download dos pesos. A inferência real ONNX e a interface precisam de validação no Windows com as dependências instaladas.
