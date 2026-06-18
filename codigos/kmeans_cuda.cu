/**
 * @file kmeans_cuda.cu
 * @brief K Means Clustering - versao CUDA para GPU
 * @details
 * Paralelizacao com CUDA. Cada thread processa um ponto de dados.
 * Usa atomicAdd para acumulacao dos centróides e reducao para contar mudancas.
 * @author [Lakhan Nad](https://github.com/Lakhan-Nad) — paralelizacao CUDA por [grupo]
 */

/* TEMPOS DE EXECUCAO (preencher apos medicoes):
 * Dataset: NYC Taxi 10M pontos, K=7

    Devices CUDA: 1
    GPU: GeForce GT 1030 (Compute 6.1)
    Carregando dados de 'data.csv'...
    Pontos carregados  : 7000000
    Clusters (k)       : 11
    Tempo kMeans (CUDA GPU): 4.8877 s

    real    0m12.847s
    user    0m10.866s
    sys     0m1.898s
    Speedup: 4,8773 = 23.8389 / 4.8877
 */

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define MAX_POINTS 7000000
#define BLOCK_SIZE 256

/* MODIFICACAO 10: atomicAdd para double nao existe antes de compute 6.0.
 * Implementacao via atomicCAS (compare-and-swap) funciona em qualquer GPU.
 */
__device__ double atomicAddDouble(double *address, double val)
{
    unsigned long long int *addr_as_ull = (unsigned long long int *)address;
    unsigned long long int  old         = *addr_as_ull;
    unsigned long long int  assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

/* Macro para checar erros CUDA */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

typedef struct observation {
    double x;
    double y;
    int    group;
} observation;

typedef struct cluster {
    double x;
    double y;
    size_t count;
} cluster;

/* ==========================================================================
 * MODIFICACAO 1: Kernel que zera os centróides no device.
 * Um thread por cluster — loop pequeno, basta 1 bloco.
 * ========================================================================== */
__global__ void resetCentroids(double *cx, double *cy, int *ccount, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k)
    {
        cx[i]     = 0.0;
        cy[i]     = 0.0;
        ccount[i] = 0;
    }
}

/* ==========================================================================
 * MODIFICACAO 2: Kernel de acumulacao com shared memory.
 * Cada bloco acumula em memoria compartilhada local (s_cx, s_cy, s_count),
 * depois faz apenas 1 atomic global por cluster por bloco.
 * Reduz contencao de 7M atomics globais para ~27K (7M / BLOCK_SIZE).
 * ========================================================================== */
__global__ void accumulateCentroids(const double *ox, const double *oy,
                                    const int *ogrp,
                                    double *cx, double *cy, int *ccount,
                                    int n, int k)
{
    /* Shared memory dinamica: k doubles para x, k doubles para y, k ints para count */
    extern __shared__ char smem[];
    double *s_cx    = (double *)smem;
    double *s_cy    = s_cx + k;
    int    *s_count = (int *)(s_cy + k);

    /* Inicializa shared memory */
    for (int i = threadIdx.x; i < k; i += blockDim.x)
    {
        s_cx[i]    = 0.0;
        s_cy[i]    = 0.0;
        s_count[i] = 0;
    }
    __syncthreads();

    /* Cada thread acumula seu ponto na shared memory do bloco */
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n)
    {
        int g = ogrp[j];
        atomicAddDouble(&s_cx[g], ox[j]);   /* atomic em shared -- muito rapido */
        atomicAddDouble(&s_cy[g], oy[j]);
        atomicAdd(&s_count[g], 1);
    }
    __syncthreads();

    /* Reduz resultados do bloco no vetor global -- apenas k atomics por bloco */
    for (int i = threadIdx.x; i < k; i += blockDim.x)
    {
        if (s_count[i] > 0)
        {
            atomicAddDouble(&cx[i], s_cx[i]);
            atomicAddDouble(&cy[i], s_cy[i]);
            atomicAdd(&ccount[i], s_count[i]);
        }
    }
}

/* ==========================================================================
 * MODIFICACAO 3: Kernel de normalizacao dos centróides.
 * Um thread por cluster.
 * ========================================================================== */
__global__ void normalizeCentroids(double *cx, double *cy,
                                   const int *ccount, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k && ccount[i] > 0)
    {
        cx[i] /= (double)ccount[i];
        cy[i] /= (double)ccount[i];
    }
}

/* ==========================================================================
 * MODIFICACAO 4: Kernel de re-atribuicao dos pontos ao centróide mais proximo.
 * Um thread por ponto. Usa atomicAdd em d_changed para contar mudancas.
 * ========================================================================== */
__global__ void assignPoints(const double *ox, const double *oy,
                             int *ogrp,
                             const double *cx, const double *cy,
                             int k, int n, int *d_changed)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    double minD    = DBL_MAX;
    int    nearest = 0;

    for (int i = 0; i < k; i++)
    {
        double dx   = cx[i] - ox[j];
        double dy   = cy[i] - oy[j];
        double dist = dx * dx + dy * dy;
        if (dist < minD)
        {
            minD    = dist;
            nearest = i;
        }
    }

    if (nearest != ogrp[j])
    {
        ogrp[j] = nearest;
        atomicAdd(d_changed, 1);
    }
}

/* ==========================================================================
 * Funcao principal do K-Means com CUDA
 * ========================================================================== */
cluster *kMeans(observation observations[], size_t size, int k)
{
    cluster *clusters = (cluster *)malloc(sizeof(cluster) * k);
    memset(clusters, 0, k * sizeof(cluster));

    /* STEP 1: atribuicao aleatoria inicial (na CPU) */
    for (size_t j = 0; j < size; j++)
        observations[j].group = rand() % k;

    int n = (int)size;

    /* MODIFICACAO 5: Converte AoS para SoA para transferencia eficiente.
     * Arrays planos sao mais eficientes para coalescing de memoria na GPU.
     */
    double *h_ox   = (double *)malloc(n * sizeof(double));
    double *h_oy   = (double *)malloc(n * sizeof(double));
    int    *h_ogrp = (int *)   malloc(n * sizeof(int));

    for (int j = 0; j < n; j++)
    {
        h_ox[j]   = observations[j].x;
        h_oy[j]   = observations[j].y;
        h_ogrp[j] = observations[j].group;
    }

    /* MODIFICACAO 6: Aloca memoria no device (GPU).
     * d_ prefixo indica ponteiro no device.
     */
    double *d_ox, *d_oy, *d_cx, *d_cy;
    int    *d_ogrp, *d_ccount, *d_changed;

    CUDA_CHECK(cudaMalloc(&d_ox,      n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_oy,      n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ogrp,    n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cx,      k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cy,      k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ccount,  k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    /* MODIFICACAO 7: Transfere dados de entrada para o device uma unica vez. */
    CUDA_CHECK(cudaMemcpy(d_ox,   h_ox,   n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oy,   h_oy,   n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ogrp, h_ogrp, n * sizeof(int),    cudaMemcpyHostToDevice));

    int blocks_n = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_k = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int changed          = 0;
    int minAcceptedError = n / 10000;

    /* Loop principal do K-Means */
    do
    {
        /* STEP 2a: zera centróides no device */
        resetCentroids<<<blocks_k, BLOCK_SIZE>>>(d_cx, d_cy, d_ccount, k);

        /* STEP 2b: acumula contribuicoes de cada ponto */
        /* smem: k*2 doubles + k ints por bloco */
        size_t smem_size = k * (2 * sizeof(double) + sizeof(int));
        accumulateCentroids<<<blocks_n, BLOCK_SIZE, smem_size>>>(d_ox, d_oy, d_ogrp,
                                                                   d_cx, d_cy, d_ccount, n, k);

        /* STEP 2c: normaliza centróides */
        normalizeCentroids<<<blocks_k, BLOCK_SIZE>>>(d_cx, d_cy, d_ccount, k);

        /* STEP 3+4: re-atribuicao — zera contador de mudancas no device */
        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));

        assignPoints<<<blocks_n, BLOCK_SIZE>>>(d_ox, d_oy, d_ogrp,
                                               d_cx, d_cy, k, n, d_changed);

        /* Copia numero de mudancas de volta ao host para checar convergencia */
        CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));

    } while (changed > minAcceptedError);

    /* MODIFICACAO 8: Copia resultados finais do device para o host. */
    CUDA_CHECK(cudaMemcpy(h_ogrp, d_ogrp, n * sizeof(int), cudaMemcpyDeviceToHost));

    /* Recalcula centróides finais na CPU para preencher clusters[] */
    for (int j = 0; j < n; j++)
        observations[j].group = h_ogrp[j];

    for (int j = 0; j < n; j++)
    {
        int g = h_ogrp[j];
        clusters[g].x += h_ox[j];
        clusters[g].y += h_oy[j];
        clusters[g].count++;
    }
    for (int i = 0; i < k; i++)
        if (clusters[i].count > 0)
        {
            clusters[i].x /= clusters[i].count;
            clusters[i].y /= clusters[i].count;
        }

    /* Libera memoria do device e host */
    cudaFree(d_ox); cudaFree(d_oy); cudaFree(d_ogrp);
    cudaFree(d_cx); cudaFree(d_cy); cudaFree(d_ccount);
    cudaFree(d_changed);
    free(h_ox); free(h_oy); free(h_ogrp);

    return clusters;
}

/* ==========================================================================
 * Funcao loadCSV: identica as demais versoes
 * ========================================================================== */
size_t loadCSV(const char *filename, observation *observations, size_t maxPoints)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Erro: nao foi possivel abrir '%s'\n", filename); exit(1); }
    char line[128];
    size_t count = 0;
    fgets(line, sizeof(line), fp); /* pula cabecalho */
    while (count < maxPoints && fgets(line, sizeof(line), fp))
    {
        double lon, lat;
        if (sscanf(line, "%lf,%lf", &lon, &lat) == 2)
            if (lon != 0.0 && lat != 0.0)
            {
                observations[count].x     = lon;
                observations[count].y     = lat;
                observations[count].group = 0;
                count++;
            }
    }
    fclose(fp);
    return count;
}

int main(int argc, char *argv[])
{
    const char *filename = (argc > 1) ? argv[1] : "data.csv";
    int k = 11;

    srand(42);

    /* Mostra informacoes do device CUDA */
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Devices CUDA: %d\n", deviceCount);
    if (deviceCount > 0)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("GPU: %s (Compute %d.%d)\n",
               prop.name, prop.major, prop.minor);
    }

    printf("Carregando dados de '%s'...\n", filename);

    observation *observations = (observation *)malloc(sizeof(observation) * MAX_POINTS);
    if (!observations) { fprintf(stderr, "Erro: malloc falhou\n"); return 1; }

    size_t size = loadCSV(filename, observations, MAX_POINTS);
    printf("Pontos carregados  : %zu\n", size);
    printf("Clusters (k)       : %d\n",  k);

    /* MODIFICACAO 9: cudaEvent para medicao de tempo precisa na GPU */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cluster *clusters = kMeans(observations, size, k);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Tempo kMeans (CUDA GPU): %.4f s\n", ms / 1000.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(observations);
    free(clusters);
    return 0;
}
