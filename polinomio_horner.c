#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>   /* Libreria de Intel que permite usar instrucciones vectoriales (AVX) */

#define N 10000           /* Cantidad de coeficientes. Debe ser multiplo de 8 porque AVX procesa de 8 en 8 */
#define NUM_TRIALS 100000 /* Repetimos la evaluacion muchas veces para que el tiempo medido sea confiable */

/* ---------------------------------------------------------------
   VERSION NORMAL (escalar)
   Evalua un polinomio con el metodo de Horner. En vez de calcular
   cada potencia (X^2, X^3, ...) por separado, va acumulando:
   suma un coeficiente, multiplica por X, y repite.
   Asi se necesitan solo N sumas y N multiplicaciones.
   --------------------------------------------------------------- */
float horner(float X, const float *coef, long size)
{
    float ACC = 0.0f;
    long i;
    for (i = 0; i < size; i++) {
        ACC = (ACC + coef[i]) * X;   /* Esta es la formula de Horner */
    }
    return ACC;
}

/* ---------------------------------------------------------------
   VERSION RAPIDA (con AVX)
   Idea: un registro __m256 es como una "caja" con 8 floats que la
   CPU puede sumar o multiplicar TODOS A LA VEZ en una sola
   instruccion. En lugar de un Horner, hacemos 8 Horner en paralelo.

   Cada casilla de la caja (carril) toma un coeficiente de cada 8:
       carril 0 -> coef[0], coef[8],  coef[16], ...
       carril 1 -> coef[1], coef[9],  coef[17], ...
       ... y asi hasta el carril 7.
   Como cada carril salta de 8 en 8, en cada paso se multiplica por
   X^8 (no por X).
   --------------------------------------------------------------- */
float horner_intrinsic(float X, const float *coef, long size)
{
    float R[8] __attribute__((aligned(32)));  /* Aqui copiaremos las 8 casillas para leerlas una por una */
    float X2 = X * X;
    float X4 = X2 * X2;
    float X8 = X4 * X4;                       /* X^8: el factor que usa cada carril en cada paso */
    long i;
    long bloques = size / 8;                  /* Cuantos grupos de 8 coeficientes hay */

    __m256 X256 = _mm256_set1_ps(X8);         /* Caja con X^8 copiado en las 8 casillas */
    __m256 Y    = _mm256_setzero_ps();        /* Caja acumuladora: 8 acumuladores, todos en 0 */

    for (i = 0; i < bloques - 1; i++) {
        /* Carga 8 coeficientes de golpe y los suma a los 8 acumuladores.
           La memoria debe estar "alineada" (ver _mm_malloc en main), si no el programa falla. */
        Y = _mm256_add_ps(Y, _mm256_load_ps(coef + 8 * i));
        Y = _mm256_mul_ps(Y, X256);           /* Multiplica las 8 casillas por X^8 a la vez */
    }

    /* El ultimo grupo solo se suma, sin multiplicar por X^8, porque cada casilla
       necesita una potencia final distinta. Eso se arregla en el paso siguiente. */
    Y = _mm256_add_ps(Y, _mm256_load_ps(coef + 8 * i));

    _mm256_store_ps(R, Y);                    /* Pasamos las 8 casillas a un arreglo normal */

    /* Unimos las 8 casillas en un solo numero. Cada una debe multiplicarse
       por la potencia de X que le toca:
           casilla 7 -> X^1,  casilla 6 -> X^2,  ...,  casilla 0 -> X^8 */
    float P = 0.0f;
    float pot = X;                            /* Empieza en X^1 */
    for (int k = 7; k >= 0; k--) {
        P += R[k] * pot;
        pot *= X;                             /* Sube a la siguiente potencia */
    }
    return P;                                 /* Mismo resultado que horner(), pero mas rapido */
}

int main(void)
{
    /* X cercano a 1 a proposito: un float solo llega hasta ~3.4e38.
       Con X = 1.1, el resultado seria 1.1^10000 (enorme) y saldria "inf". */
    float X = 1.0001f;
    float R;
    volatile float sink = 0.0f;  /* Truco: obliga al compilador a ejecutar de verdad los ciclos
                                    de medicion en vez de eliminarlos por "no servir para nada" */
    int i, j;
    clock_t t1, t2;
    float diff;

    srand((unsigned)time(NULL));

    /* Reserva memoria alineada a 32 bytes. AVX lo exige para cargar 8 floats de una vez.
       Por eso se usa _mm_malloc y despues _mm_free (no malloc/free normales). */
    float *coeficientes = (float *)_mm_malloc(N * sizeof(float), 32);
    if (!coeficientes) {
        fprintf(stderr, "No se pudo reservar memoria\n");
        return 1;
    }

    /* Coeficientes aleatorios entre 0 y 0.999 */
    printf("Primeros 10 coeficientes:\n");
    for (i = 0; i < N; i++) {
        coeficientes[i] = (float)(rand() % 1000) / 1000.0f;
        if (i < 10)
            printf("  coef[%d] = %f\n", i, coeficientes[i]);
    }

    /* Primero comprobamos que las dos versiones den (casi) el mismo resultado.
       Pueden diferir en los ultimos digitos por el redondeo de float. */
    R = horner(X, coeficientes, N);
    printf("Resultado Horner escalar:  %f\n", R);
    R = horner_intrinsic(X, coeficientes, N);
    printf("Resultado Horner con AVX:  %f\n", R);

    /* Medimos cuanto tarda la version normal */
    t1 = clock();
    for (j = 0; j < NUM_TRIALS; j++) {
        R = horner(X, coeficientes, N);
        sink += R;
    }
    t2 = clock();
    diff = (float)(t2 - t1) / CLOCKS_PER_SEC;
    printf("Tiempo Horner escalar: %f segundos\n", diff);

    /* Medimos cuanto tarda la version AVX y comparamos */
    t1 = clock();
    for (j = 0; j < NUM_TRIALS; j++) {
        R = horner_intrinsic(X, coeficientes, N);
        sink += R;
    }
    t2 = clock();
    diff = (float)(t2 - t1) / CLOCKS_PER_SEC;
    printf("Tiempo Horner con AVX: %f segundos\n", diff);

    _mm_free(coeficientes);   /* Liberar la memoria reservada con _mm_malloc */
    return 0;
}
