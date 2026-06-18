Luiz Fernando Antunes da Silva Frassi
================================================================================
Pedro Augusto Gomes
================================================================================
  K-MEANS CLUSTERING — PARALELIZACAO COM OpenMP E CUDA
  Disciplina: Programacao Paralela
================================================================================

--------------------------------------------------------------------------------
1. DESCRICAO DA APLICACAO
--------------------------------------------------------------------------------

O algoritmo K-Means e um metodo de aprendizado de maquina nao supervisionado
que agrupa N pontos de dados em K clusters. O algoritmo funciona da seguinte
forma:

  1. Atribui cada ponto aleatoriamente a um dos K clusters (inicializacao).
  2. Calcula o centroide (media) de cada cluster.
  3. Reatribui cada ponto ao cluster cujo centroide e o mais proximo.
  4. Repete os passos 2 e 3 ate que menos de 0,01% dos pontos mudem de cluster
     (criterio de convergencia).

O codigo base foi obtido em:
  https://github.com/TheAlgorithms/C/tree/master

Dataset utilizado: NYC Yellow Taxi Trip Data (Janeiro 2015)
  - Fonte: https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data?select=yellow_tripdata_2015-01.csv
  - Arquivo original: yellow_tripdata_2015-01.csv
  - Colunas usadas: pickup_longitude e pickup_latitude
  - Apos pre-processamento: ~10 milhoes de pontos validos
  - Clusters: K = 11

--------------------------------------------------------------------------------
2. PRE-PROCESSAMENTO DO DATASET
--------------------------------------------------------------------------------

O dataset original precisa ser pre-processado antes de executar os algoritmos.
O script Python em data/tratamento_base.py realiza essa etapa.

Requisitos:
  pip install pandas

Execucao:
  cd data/
  python tratamento_base.py

O script le o arquivo yellow_tripdata_2015-01.csv, remove coordenadas nulas
ou invalidas (0,0) e gera o arquivo data.csv com duas colunas:
  pickup_longitude, pickup_latitude

O arquivo data.csv deve estar no mesmo diretorio dos executaveis ao rodar
os algoritmos.

--------------------------------------------------------------------------------
3. VERSOES IMPLEMENTADAS
--------------------------------------------------------------------------------

  kmeans_seq.c          — Versao sequencial (codigo base adaptado)
  kmeans_openmp_cpu.c   — Versao paralela com OpenMP para CPU (multicore)
  kmeans_openmp_gpu.c   — Versao paralela com OpenMP target offload para GPU
  kmeans_cuda.cu        — Versao paralela com CUDA para GPU NVIDIA

--------------------------------------------------------------------------------
4. COMPILACAO
--------------------------------------------------------------------------------

4.1 Versao Sequencial
---------------------
  gcc -O3 -o kmeans_seq kmeans_seq.c -lm

4.2 OpenMP para CPU
-------------------
  gcc -O3 -fopenmp -o kmeans_omp_cpu kmeans_openmp_cpu.c -lm

4.3 OpenMP para GPU (target offload — NVIDIA)
---------------------------------------------
  clang -O3 -fopenmp \
      -fopenmp-targets=nvptx64-nvidia-cuda \
      --offload-arch=sm_61 \
      -o kmeans_omp_gpu kmeans_openmp_gpu.c -lm

  Ajuste o sm_XX de acordo com a compute capability da GPU disponivel.
  Exemplos: sm_61 (GT 1030), sm_70 (V100), sm_80 (A100), sm_86 (RTX 3090).

4.4 CUDA para GPU NVIDIA
------------------------
  nvcc -O3 -o kmeans_cuda kmeans_cuda.cu -lm

--------------------------------------------------------------------------------
5. EXECUCAO
--------------------------------------------------------------------------------

5.1 Versao Sequencial
---------------------
  ./kmeans_seq data.csv

5.2 OpenMP CPU — segundo argumento define numero de threads
-----------------------------------------------------------
  ./kmeans_omp_cpu data.csv 1
  ./kmeans_omp_cpu data.csv 2
  ./kmeans_omp_cpu data.csv 4
  ./kmeans_omp_cpu data.csv 8
  ./kmeans_omp_cpu data.csv 16
  ./kmeans_omp_cpu data.csv 32

  (Sem segundo argumento usa o maximo de threads disponivel)

5.3 OpenMP GPU
--------------
  ./kmeans_omp_gpu data.csv

5.4 CUDA GPU
------------
  ./kmeans_cuda data.csv

--------------------------------------------------------------------------------
6. TEMPOS DE EXECUCAO
--------------------------------------------------------------------------------

  Dataset : NYC Taxi — ~10M pontos, K=7

 Encontra-se no inicio de cada codigo.

--------------------------------------------------------------------------------
7. MUDANCAS REALIZADAS PARA PARALELIZACAO
--------------------------------------------------------------------------------

kmeans_openmp_cpu.c:
  - Acumulacao dos centróides paralelizada com reducao local por thread
    (#pragma omp parallel + critical para combinar resultados)
  - Reatribuicao dos pontos paralelizada com
    #pragma omp parallel for reduction(+:changed)
  - Medicao de tempo com omp_get_wtime()
  - Numero de threads configuravel via argumento de linha de comando

kmeans_openmp_gpu.c:
  - Dados convertidos para SoA (Structure of Arrays) para mapeamento eficiente
  - #pragma omp target data mantem arrays no device durante todo o loop
  - Zeragem e normalizacao dos centróides via #pragma omp target teams distribute
  - Acumulacao com #pragma omp atomic update no device
  - Reatribuicao com reduction(+:changed) no device
  - calculateNearst declarada com #pragma omp declare target

kmeans_cuda.cu:
  - 4 kernels CUDA: resetCentroids, accumulateCentroids,
    normalizeCentroids, assignPoints
  - Acumulacao usa shared memory por bloco para reduzir contenção de atomics
    globais de O(N) para O(N/BLOCK_SIZE)
  - atomicAdd para double implementado via atomicCAS (compativel com
    compute capability < 6.0)
  - Medicao de tempo com cudaEvent (sincronizada com a GPU)
  - Transferencia de dados ao device feita uma unica vez fora do loop

--------------------------------------------------------------------------------
8. ESTRUTURA DE DIRETORIOS
--------------------------------------------------------------------------------

  trab_final_paralela/
  |-- kmeans_seq.c              (versao sequencial)
  |-- kmeans_openmp_cpu.c       (versao OpenMP CPU)
  |-- kmeans_openmp_gpu.c       (versao OpenMP GPU)
  |-- kmeans_cuda.cu            (versao CUDA)
  |-- readme.txt                (este arquivo)
  |-- data/
      |-- tratamento_base.py    (script de pre-processamento)
      |-- yellow_tripdata_2015-01.csv  (dataset original — baixar separadamente)
      |-- data.csv              (gerado pelo script acima)

--------------------------------------------------------------------------------
9. DEPENDENCIAS
--------------------------------------------------------------------------------

  - GCC >= 9.0 com suporte a OpenMP 4.5
  - Clang com suporte a OpenMP target offload (nvptx64)
  - CUDA Toolkit >= 10.0
  - Python 3 + pandas (para pre-processamento do dataset)
  - Dataset: yellow_tripdata_2015-01.csv

================================================================================
