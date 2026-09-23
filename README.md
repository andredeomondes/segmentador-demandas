# Segmentador de Demandas por Aprendizado Não Supervisionado

Descobre os temas recorrentes em solicitações de atendimento sem usar o rótulo humano,
revelando agrupamentos que a taxonomia oficial não cobre.

O rótulo existe no corpus, mas não participa do treino nem da escolha de parâmetros. É lido uma
única vez, ao final, para quantificar a discordância.

## Resultado

O modelo escolheu k=7 sozinho, exatamente o número de categorias humanas, sem nunca tê-las
visto.

| Grupo | Docs | Termos definidores | Categoria dominante | Pureza |
|---|---:|---|---|---:|
| 4 | 20 | cancel, renov, quer cancel | cancelamento | 95% |
| 6 | 5 | cheg, praz, pacot | entrega | 80% |
| 2 | 8 | empr, canal, exist | outros | 62% |
| 3 | 10 | sab, gost sab, func | outros | 60% |
| 1 | 19 | cont, acess, vincul, esquec | acesso | 53% |
| 5 | 18 | receb, cobrança, produt, nunc | fraude | 28% |
| 0 | 60 | aplic, entreg, ped, pag | problema_tecnico | 28% |

Índice de Rand ajustado: 0,181, concordância fraca com a taxonomia humana.

## A leitura que importa

Rand baixo não significa que a segmentação falhou. Significa que o modelo dividiu o corpus por
outro critério. Olhando grupo a grupo, a discordância tem estrutura:

- `cancelamento` e `entrega` são categorias reais. O modelo as isolou com 95% e 80% de
  pureza sem ajuda nenhuma. São temas linguisticamente distintos, com vocabulário próprio.
- `pagamento`, `fraude` e `problema_tecnico` se sobrepõem de fato. O grupo 0 absorveu 60
  dos 140 documentos misturando as três. Não há vocabulário que as separe.

Esse mesmo achado aparece de forma independente no
[classificador supervisionado](https://github.com/andredeomondes/classificador-triagem-atendimento),
onde os erros se concentram exatamente nessa fronteira.

Dois métodos distintos, um supervisionado e outro não, apontaram o mesmo problema: a taxonomia
mistura categorias que o texto não separa. Esse resultado é mais acionável do que
qualquer métrica alta, sugere revisar os rótulos, não trocar o modelo.

## Método

| Etapa | Escolha | Justificativa |
|---|---|---|
| Pré-processamento | NLTK: tokenização, stopwords, stemmer RSLP | RSLP é projetado para a morfologia do português |
| Representação | TF-IDF, unigrama e bigrama, `min_df=2` | Termo visto em um único documento não forma grupo |
| Agrupamento | K-Means | |
| Escolha do k | Silhueta, testando k de 2 a 10 | Ver abaixo |
| Comparação final | Índice de Rand ajustado | Lido apenas no fim |

### Por que a silhueta escolhe o k

Num problema real de segmentação não existe gabarito dizendo quantos grupos há. Usar o número
de categorias humanas seria contaminar a escolha com a resposta.

A silhueta mede coesão interna contra separação entre grupos usando apenas a geometria dos
vetores. É um critério honesto, porque não olha o rótulo. Que ele tenha convergido para k=7,
coincidindo com a taxonomia humana, é resultado, não premissa.

## Limitações

- A silhueta ficou baixa em todos os k testados (0,019 a 0,044). Com 140 documentos curtos
  em espaço TF-IDF esparso, os grupos não têm separação geométrica forte. O número reportado é
  o que o dado permite, não o que seria conveniente.
- K-Means assume grupos esféricos e de tamanho semelhante. O grupo 0, com 60 documentos
  contra 5 do grupo 6, mostra que a premissa não se sustenta aqui.
- TF-IDF não captura sinônimo. "Estorno" e "reembolso" ocupam dimensões distintas, ainda
  que signifiquem o mesmo.
- Corpus pequeno e sintético, com classes artificialmente balanceadas.

## Arquitetura

```
src/segmenter/
├── dataset.py        carregamento e propriedades do corpus
├── preprocessing.py  estratégias de normalização (protocolo + implementações)
├── vectorization.py  construção da matriz documento-termo
├── clustering.py     seleção de k, agrupamento e perfil dos grupos
├── reporting.py      saída em console e figuras
└── __main__.py       interface de linha de comando
```

`Preprocessor` é um `Protocol`: trocar a estratégia de normalização não exige alterar o
agrupamento, e os testes injetam implementações próprias sem tocar no código de produção.

## Instalação e uso

```bash
pip install -e ".[dev]"
python -m segmenter
```

Opções:

```bash
python -m segmenter --min-k 3 --max-k 15 --output figuras/
```

## Testes

```bash
pytest
ruff check .
```

São 11 testes, incluindo a verificação de que o k selecionado de fato maximiza a silhueta e de que
um corpus com separação óbvia produz concordância perfeita, garantindo que o índice de Rand
baixo no corpus real é propriedade do dado, não defeito do código.

## Próximo passo

Substituir TF-IDF por embeddings de sentença, que capturam sinônimo, e verificar se o índice de
Rand sobe. Isso testa se a discordância vem da representação ou da taxonomia. Hoje as duas hipóteses
seguem abertas.
