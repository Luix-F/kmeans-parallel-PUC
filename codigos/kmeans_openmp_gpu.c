/**
 * @file kmeans_openmp_gpu.c
 * @brief K Means Clustering - versao OpenMP para GPU (target offload)
 * @details
 * Paralelizacao com OpenMP target offload (diretivas #pragma omp target).
 * Os dados sao mapeados para a GPU uma unica vez e os kernels rodam no device.
 * @author [Lakhan Nad](https://github.com/Lakhan-Nad) — paralelizacao GPU por [grupo]
 */

/* TEMPOS DE EXECUCAO (preencher apos medicoes):
 * Dataset: NYC Taxi 10M pontos, K=7
    Clusters (k)       : 11
    Tempo kMeans (OpenMP GPU): 2.2584 s

    real    0m11.299s
    user    0m8.196s
    sys     0m0.772s
    Speedup: 10,5556 = 23.8389 / 2.2584
 */

#define _USE_MATH_DEFINES
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>          /* MODIFICACAO 1: OpenMP com suporte a target */

#define MAX_POINTS 7000000

typedef struct observation
{
    double x;
    double y;
    int group;
} observation;

typedef struct cluster
{
    double x;
    double y;
    size_t count;
} cluster;

/* MODIFICACAO 2: calculateNearst declarada com omp declare target
 * para que possa ser chamada dentro de regioes target na GPU.
 */
#pragma omp declare target
int calculateNearst(double ox, double oy, double *cx, double *cy, int k)
{
    double minD = DBL_MAX;
    int index = 0;
    for (int i = 0; i < k; i++)
    {
        double dx = cx[i] - ox;
        double dy = cy[i] - oy;
        double dist = dx * dx + dy * dy;
        if (dist < minD)
        {
            minD = dist;
            index = i;
        }
    }
    return index;
}
#pragma omp end declare target

cluster *kMeans(observation observations[], size_t size, int k)
{
    cluster *clusters = NULL;

    if (k <= 1)
    {
        clusters = (cluster *)malloc(sizeof(cluster));
        memset(clusters, 0, sizeof(cluster));
        /* caso trivial: sem offload */
        double sx = 0, sy = 0;
        for (size_t i = 0; i < size; i++) { sx += observations[i].x; sy += observations[i].y; }
        clusters[0].x = sx / size;
        clusters[0].y = sy / size;
        clusters[0].count = size;
        return clusters;
    }

    if (k >= (int)size)
    {
        clusters = (cluster *)malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        for (int j = 0; j < (int)size; j++)
        {
            clusters[j].x = observations[j].x;
            clusters[j].y = observations[j].y;
            clusters[j].count = 1;
            observations[j].group = j;
        }
        return clusters;
    }

    clusters = (cluster *)malloc(sizeof(cluster) * k);
    memset(clusters, 0, k * sizeof(cluster));

    /* STEP 1: atribuicao aleatoria inicial (na CPU) */
    for (size_t j = 0; j < size; j++)
        observations[j].group = rand() % k;

    /* MODIFICACAO 3: arrays separados de SoA para facilitar o mapeamento GPU.
     * x[], y[], group[] extraidos de observations[] para uso no device.
     * cx[], cy[], ccount[] sao os centróides mapeados no device.
     */
    double *ox    = (double *)malloc(size * sizeof(double));
    double *oy    = (double *)malloc(size * sizeof(double));
    int    *ogrp  = (int *)   malloc(size * sizeof(int));
    double *cx    = (double *)malloc(k    * sizeof(double));
    double *cy    = (double *)malloc(k    * sizeof(double));
    size_t *ccount= (size_t *)malloc(k    * sizeof(size_t));

    for (size_t j = 0; j < size; j++)
    {
        ox[j]   = observations[j].x;
        oy[j]   = observations[j].y;
        ogrp[j] = observations[j].group;
    }

    size_t changed         = 0;
    size_t minAcceptedError = size / 10000;

    /* MODIFICACAO 4: mapa de dados para GPU com target data.
     * ox[], oy[] sao somente leitura (map to).
     * ogrp[], cx[], cy[], ccount[] precisam de leitura e escrita (map tofrom).
     * O mapa e mantido durante todo o loop do-while para evitar
     * transferencias repetidas a cada iteracao.
     */
    #pragma omp target data \
        map(to:     ox[0:size], oy[0:size]) \
        map(tofrom: ogrp[0:size]) \
        map(alloc:  cx[0:k], cy[0:k], ccount[0:k])
    {
        do
        {
            /* ----------------------------------------------------------
             * STEP 2a: zera centróides no device
             * MODIFICACAO 5: target teams distribute para loop pequeno (k).
             * ---------------------------------------------------------- */
            #pragma omp target teams distribute parallel for
            for (int i = 0; i < k; i++)
            {
                cx[i]     = 0.0;
                cy[i]     = 0.0;
                ccount[i] = 0;
            }

            /* ----------------------------------------------------------
             * STEP 2b: acumulacao — cada ponto soma sua contribuicao
             * ao centróide do seu grupo.
             * MODIFICACAO 6: atomic garante ausencia de race condition
             * na acumulacao paralela no device.
             * ---------------------------------------------------------- */
            #pragma omp target teams distribute parallel for
            for (size_t j = 0; j < size; j++)
            {
                int g = ogrp[j];
                #pragma omp atomic update
                cx[g] += ox[j];
                #pragma omp atomic update
                cy[g] += oy[j];
                #pragma omp atomic update
                ccount[g] += (size_t)1;
            }

            /* ----------------------------------------------------------
             * STEP 2c: normaliza centróides (divisao pelo count).
             * MODIFICACAO 7: loop pequeno sobre k clusters no device.
             * ---------------------------------------------------------- */
            #pragma omp target teams distribute parallel for
            for (int i = 0; i < k; i++)
            {
                if (ccount[i] > 0)
                {
                    cx[i] /= (double)ccount[i];
                    cy[i] /= (double)ccount[i];
                }
            }

            /* ----------------------------------------------------------
             * STEP 3+4: re-atribuicao de cada ponto ao centróide mais proximo.
             * MODIFICACAO 8: reduction(+:changed) acumula quantos pontos
             * mudaram de grupo sem race condition.
             * cx[] e cy[] sao lidos como firstprivate implicito via map alloc
             * (ja estao no device desde o map externo).
             * ---------------------------------------------------------- */
            changed = 0;
            #pragma omp target teams distribute parallel for \
                reduction(+:changed)
            for (size_t j = 0; j < size; j++)
            {
                int nearest = calculateNearst(ox[j], oy[j], cx, cy, k);
                if (nearest != ogrp[j])
                {
                    changed++;
                    ogrp[j] = nearest;
                }
            }

        } while (changed > minAcceptedError);

    } /* fim do target data — dados copiados de volta automaticamente */

    /* Copia grupos de volta para observations[] e monta clusters[] */
    for (size_t j = 0; j < size; j++)
        observations[j].group = ogrp[j];

    /* Recalcula centróides finais na CPU para preencher clusters[] */
    memset(clusters, 0, k * sizeof(cluster));
    for (size_t j = 0; j < size; j++)
    {
        int g = ogrp[j];
        clusters[g].x += ox[j];
        clusters[g].y += oy[j];
        clusters[g].count++;
    }
    for (int i = 0; i < k; i++)
        if (clusters[i].count > 0)
        {
            clusters[i].x /= clusters[i].count;
            clusters[i].y /= clusters[i].count;
        }

    free(ox); free(oy); free(ogrp);
    free(cx); free(cy); free(ccount);

    return clusters;
}

/* Funcao loadCSV: identica as versoes sequencial e CPU */
size_t loadCSV(const char *filename, observation *observations, size_t maxPoints)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        fprintf(stderr, "Erro: nao foi possivel abrir '%s'\n", filename);
        exit(1);
    }
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

    printf("Carregando dados de '%s'...\n", filename);

    observation *observations =
        (observation *)malloc(sizeof(observation) * MAX_POINTS);
    if (!observations) { fprintf(stderr, "Erro: malloc falhou\n"); return 1; }

    size_t size = loadCSV(filename, observations, MAX_POINTS);
    printf("Pontos carregados  : %zu\n", size);
    printf("Clusters (k)       : %d\n",  k);

    /* MODIFICACAO 9: omp_get_wtime() para medicao de wall time */
    double t_start = omp_get_wtime();

    cluster *clusters = kMeans(observations, size, k);

    double elapsed = omp_get_wtime() - t_start;

    printf("Tempo kMeans (OpenMP GPU): %.4f s\n", elapsed);

    free(observations);
    free(clusters);
    return 0;
}