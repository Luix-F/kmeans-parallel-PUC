/**
 * @file k_means_sequential.c
 * @brief K Means Clustering Algorithm implemented
 * @details
 * This file has K Means algorithm implemmented
 * It prints test output in eps format
 *
 * Note:
 * Though the code for clustering works for all the
 * 2D data points and can be extended for any size vector
 * by making the required changes, but note that
 * the output method i.e. printEPS is only good for
 * polar data points i.e. in a circle and both test
 * use the same.
 * @author [Lakhan Nad](https://github.com/Lakhan-Nad)
 */

/* TEMPOS DE EXECUCAO (preencher apos medicoes):
 * Dataset: NYC Taxi 10M pontos, K=7
    Pontos carregados: 7000000
    Clusters (k): 11
    Tempo kMeans (sequencial): 23.8389 s

    real    0m30.602s
    user    0m30.345s
    sys     0m0.232s
 */

#define _USE_MATH_DEFINES /* required for MS Visual C */
#include <float.h>        /* DBL_MAX, DBL_MIN */
#include <math.h>         /* PI, sin, cos */
#include <stdio.h>        /* printf */
#include <stdlib.h>       /* rand */
#include <string.h>       /* memset */
#include <time.h>         /* time, clock_gettime -- MODIFICACAO 1 */

/* Tamanho maximo de pontos a carregar do CSV */
#define MAX_POINTS 7000000

/*!
 * @addtogroup machine_learning Machine Learning Algorithms
 * @{
 * @addtogroup k_means K-Means Clustering Algorithm
 * @{
 */

/*! @struct observation
 *  a class to store points in 2d plane
 *  the name observation is used to denote
 *  a random point in plane
 */
typedef struct observation
{
    double x;  /**< abscissa of 2D data point */
    double y;  /**< ordinate of 2D data point */
    int group; /**< the group no in which this observation would go */
} observation;

/*! @struct cluster
 *  this class stores the coordinates
 *  of centroid of all the points
 *  in that cluster it also
 *  stores the count of observations
 *  belonging to this cluster
 */
typedef struct cluster
{
    double x;     /**< abscissa centroid of this cluster */
    double y;     /**< ordinate of centroid of this cluster */
    size_t count; /**< count of observations present in this cluster */
} cluster;

/*!
 * Returns the index of centroid nearest to
 * given observation
 *
 * @param o  observation
 * @param clusters  array of cluster having centroids coordinates
 * @param k  size of clusters array
 *
 * @returns the index of nearest centroid for given observation
 */
int calculateNearst(observation *o, cluster clusters[], int k)
{
    double minD = DBL_MAX;
    double dist = 0;
    int index = -1;
    int i = 0;
    for (; i < k; i++)
    {
        /* Calculate Squared Distance*/
        dist = (clusters[i].x - o->x) * (clusters[i].x - o->x) +
               (clusters[i].y - o->y) * (clusters[i].y - o->y);
        if (dist < minD)
        {
            minD = dist;
            index = i;
        }
    }
    return index;
}

/*!
 * Calculate centoid and assign it to the cluster variable
 *
 * @param observations  an array of observations whose centroid is calculated
 * @param size  size of the observations array
 * @param centroid  a reference to cluster object to store information of
 * centroid
 */
void calculateCentroid(observation observations[], size_t size,
                       cluster *centroid)
{
    size_t i = 0;
    centroid->x = 0;
    centroid->y = 0;
    centroid->count = size;
    for (; i < size; i++)
    {
        centroid->x += observations[i].x;
        centroid->y += observations[i].y;
        observations[i].group = 0;
    }
    centroid->x /= centroid->count;
    centroid->y /= centroid->count;
}

/*!
 *    --K Means Algorithm--
 * 1. Assign each observation to one of k groups
 *    creating a random initial clustering
 * 2. Find the centroid of observations for each
 *    cluster to form new centroids
 * 3. Find the centroid which is nearest for each
 *    observation among the calculated centroids
 * 4. Assign the observation to its nearest centroid
 *    to create a new clustering.
 * 5. Repeat step 2,3,4 until there is no change
 *    the current clustering and is same as last
 *    clustering.
 *
 * @param observations  an array of observations to cluster
 * @param size  size of observations array
 * @param k  no of clusters to be made
 *
 * @returns pointer to cluster object
 */
cluster *kMeans(observation observations[], size_t size, int k)
{
    cluster *clusters = NULL;
    if (k <= 1)
    {
        /*
        If we have to cluster them only in one group
        then calculate centroid of observations and
        that will be a ingle cluster
        */
        clusters = (cluster *)malloc(sizeof(cluster));
        memset(clusters, 0, sizeof(cluster));
        calculateCentroid(observations, size, clusters);
    }
    else if (k < size)
    {
        clusters = malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        /* STEP 1 */
        for (size_t j = 0; j < size; j++)
        {
            observations[j].group = rand() % k;
        }
        size_t changed = 0;
        size_t minAcceptedError =
            size /
            10000; // Do until 99.99 percent points are in correct cluster
        int t = 0;
        do
        {
            /* Initialize clusters */
            for (int i = 0; i < k; i++)
            {
                clusters[i].x = 0;
                clusters[i].y = 0;
                clusters[i].count = 0;
            }
            /* STEP 2*/
            for (size_t j = 0; j < size; j++)
            {
                t = observations[j].group;
                clusters[t].x += observations[j].x;
                clusters[t].y += observations[j].y;
                clusters[t].count++;
            }
            for (int i = 0; i < k; i++)
            {
                clusters[i].x /= clusters[i].count;
                clusters[i].y /= clusters[i].count;
            }
            /* STEP 3 and 4 */
            changed = 0; // this variable stores change in clustering
            for (size_t j = 0; j < size; j++)
            {
                t = calculateNearst(observations + j, clusters, k);
                if (t != observations[j].group)
                {
                    changed++;
                    observations[j].group = t;
                }
            }
        } while (changed > minAcceptedError); // Keep on grouping until we have
                                              // got almost best clustering
    }
    else
    {
        /* If no of clusters is more than observations
           each observation can be its own cluster
        */
        clusters = (cluster *)malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        for (int j = 0; j < size; j++)
        {
            clusters[j].x = observations[j].x;
            clusters[j].y = observations[j].y;
            clusters[j].count = 1;
            observations[j].group = j;
        }
    }
    return clusters;
}

/**
 * @}
 * @}
 */

/* =============================================================
 * MODIFICACAO 2: funcao loadCSV
 * Le pickup_longitude e pickup_latitude do dataset NYC Taxi.
 * Pula o cabecalho (primeira linha) e carrega ate MAX_POINTS linhas.
 * Retorna o numero de pontos carregados.
 * ============================================================= */
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

    /* Pula cabecalho */
    fgets(line, sizeof(line), fp);

    while (count < maxPoints && fgets(line, sizeof(line), fp))
    {
        double lon, lat;
        if (sscanf(line, "%lf,%lf", &lon, &lat) == 2)
        {
            /* Filtra coordenadas invalidas (zeros ou fora de NYC) */
            if (lon != 0.0 && lat != 0.0)
            {
                observations[count].x = lon;
                observations[count].y = lat;
                observations[count].group = 0;
                count++;
            }
        }
    }

    fclose(fp);
    return count;
}

/*!
 * This function calls the test
 * function
 *
 * MODIFICACAO 3: main() agora carrega CSV real, mede e imprime
 * o tempo de execucao do kMeans. printEPS removido da saida
 * para nao poluir o benchmark.
 */
int main(int argc, char *argv[])
{
    /* Arquivo CSV: pode passar como argumento ou usa padrao */
    const char *filename = (argc > 1) ? argv[1] : "data.csv";

    /* Numero de clusters */
    int k = 11;

    srand(42); /* semente fixa para reproducibilidade */

    printf("Carregando dados de '%s'...\n", filename);

    observation *observations =
        (observation *)malloc(sizeof(observation) * MAX_POINTS);
    if (!observations)
    {
        fprintf(stderr, "Erro: malloc falhou\n");
        return 1;
    }

    size_t size = loadCSV(filename, observations, MAX_POINTS);
    printf("Pontos carregados: %zu\n", size);
    printf("Clusters (k): %d\n", k);

    /* --- MEDICAO DE TEMPO (MODIFICACAO 1) --- */
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    cluster *clusters = kMeans(observations, size, k);

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    /* ---------------------------------------- */

    double elapsed = (t_end.tv_sec - t_start.tv_sec) +
                     (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    printf("Tempo kMeans (sequencial): %.4f s\n", elapsed);


    free(observations);
    free(clusters);
    return 0;
}