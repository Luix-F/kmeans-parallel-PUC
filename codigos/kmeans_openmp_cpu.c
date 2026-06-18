/**
 * @file k_means_omp_cpu.c
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

/* TEMPOS DE EXECUCAO:
 * Dataset: NYC Taxi 10M pontos, K=7

    Clusters (k)      : 11
    Threads OpenMP    : 1
    Tempo kMeans (OpenMP CPU): 23.3841 s
    real    0m30.635s
    user    0m30.391s
    Speedup: 1,0194 = 23.8389 / 23.3841


    Clusters (k)      : 11
    Threads OpenMP    : 2
    Tempo kMeans (OpenMP CPU): 12.2320 s
    real    0m19.510s
    user    0m31.336s
    sys     0m0.208s
    Speedup: 1,9488 = 23.8389 / 12.2320


    Clusters (k)      : 11
    Threads OpenMP    : 4
    Tempo kMeans (OpenMP CPU): 7.1128 s
    real    0m14.310s
    user    0m34.897s
    sys     0m0.244s
    Speedup: 3,3515 = 23.8389 / 7.1128


    Clusters (k)      : 11
    Threads OpenMP    : 8
    Tempo kMeans (OpenMP CPU): 7.5865 s
    real    0m14.763s
    user    0m34.756s
    sys     0m0.397s
    Speedup: 3,1422 = 23.8389 / 7.5865


    Clusters (k)      : 11
    Threads OpenMP    : 16
    Tempo kMeans (OpenMP CPU): 7.6100 s
    real    0m18.366s
    user    0m38.271s
    sys     0m0.533s
    Speedup: 3,1325 = 23.8389 / 7.6100


    Clusters (k)      : 11
    Threads OpenMP    : 32
    Tempo kMeans (OpenMP CPU): 7.7419 s
    real    0m14.921s
    user    0m34.987s
    sys     0m0.421s
    Speedup: 3,0792 = 23.8389 / 7.7419

 */

#define _USE_MATH_DEFINES /* required for MS Visual C */
#include <float.h>        /* DBL_MAX, DBL_MIN */
#include <math.h>         /* PI, sin, cos */
#include <stdio.h>        /* printf */
#include <stdlib.h>       /* rand */
#include <string.h>       /* memset */
#include <time.h>         /* time */
#include <omp.h>          /* MODIFICACAO 1: OpenMP */

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

/*
 * MODIFICACAO OPENMP:
 * Acumulacao local por thread para evitar race conditions.
 */
#pragma omp parallel
            {
                cluster *localClusters =
                    (cluster *)calloc(k, sizeof(cluster));

#pragma omp for
                for (size_t j = 0; j < size; j++)
                {
                    int group = observations[j].group;

                    localClusters[group].x += observations[j].x;
                    localClusters[group].y += observations[j].y;
                    localClusters[group].count++;
                }

/*
 * Combina os resultados locais no vetor global.
 */
#pragma omp critical
                {
                    for (int i = 0; i < k; i++)
                    {
                        clusters[i].x += localClusters[i].x;
                        clusters[i].y += localClusters[i].y;
                        clusters[i].count += localClusters[i].count;
                    }
                }

                free(localClusters);
            }

            for (int i = 0; i < k; i++)
            {
                if (clusters[i].count > 0)
                {
                    clusters[i].x /= clusters[i].count;
                    clusters[i].y /= clusters[i].count;
                }
            }
            /* STEP 3 and 4 */
            changed = 0; // this variable stores change in clustering
            /*
            * MODIFICACAO OPENMP:
            * Paralelizacao da atribuicao dos pontos aos clusters.
            * Reduction garante soma correta de changed.
            */
            #pragma omp parallel for reduction(+ : changed)
            for (size_t j = 0; j < size; j++)
            {
                int nearest = calculateNearst(observations + j, clusters, k);

                if (nearest != observations[j].group)
                {
                    changed++;
                    observations[j].group = nearest;
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

/* Funcao loadCSV: identica a versao sequencial */
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
 * MODIFICACAO 2+3: main aceita threads como 2o argumento.
 * Usa omp_get_wtime() para medicao de tempo de parede.
 */
int main(int argc, char *argv[])
{
    const char *filename = (argc > 1) ? argv[1] : "data.csv";
    int nthreads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    int k = 11;

    /* MODIFICACAO 2: define numero de threads antes de tudo */
    omp_set_num_threads(nthreads);

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
    printf("Pontos carregados : %zu\n", size);
    printf("Clusters (k)      : %d\n", k);
    printf("Threads OpenMP    : %d\n", nthreads);

    /* MODIFICACAO 3: omp_get_wtime() — wall time padrao OpenMP */
    double t_start = omp_get_wtime();

    cluster *clusters = kMeans(observations, size, k);

    double elapsed = omp_get_wtime() - t_start;

    printf("Tempo kMeans (OpenMP CPU): %.4f s\n", elapsed);

    free(observations);
    free(clusters);
    return 0;
}