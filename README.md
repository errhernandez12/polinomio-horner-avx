# Polinomio Horner AVX

Evaluación de un polinomio de 10 000 coeficientes con el **método de Horner**, comparando una versión escalar contra una versión vectorizada con instrucciones **AVX**. Todo está hecho con `float` (precisión simple) en C.

Proyecto del curso de Procesamiento Digital de Señales, Maestría en Comunicaciones, CINVESTAV-IPN.

## ¿Qué hace?

El programa calcula

```
P(X) = Σ coef[i] · X^(N − i)
```

de dos maneras y mide cuánto tarda cada una:

| Versión | Función | Idea |
|---|---|---|
| Escalar | `horner` | Un solo acumulador: `ACC = (ACC + coef[i]) * X` |
| AVX | `horner_intrinsic` | 8 acumuladores de Horner en paralelo dentro de un registro `__m256` |

### Cómo funciona la versión AVX

Un registro `__m256` guarda 8 `float`. Cada "carril" procesa un coeficiente de cada 8 (`coef[k]`, `coef[k+8]`, `coef[k+16]`, …), por lo que en cada paso se multiplica por **X⁸** en lugar de X. Al final, cada carril se multiplica por la potencia de X que le corresponde (carril 7 → X¹, …, carril 0 → X⁸) y se suman los 8 resultados.

Así el ciclo principal da N/8 vueltas en lugar de N.

## Requisitos

- Compilador `gcc` (o compatible) con soporte para intrínsecos AVX
- Procesador con **AVX** (casi cualquier CPU x86-64 desde 2011)

Para comprobar que tu CPU lo soporta en Linux:

```bash
grep -o -m1 avx /proc/cpuinfo
```

## Compilación y ejecución

```bash
gcc -O2 -mavx polinomio_horner.c -o polinomio_horner
./polinomio_horner
```

También puedes usar `-march=native` en lugar de `-mavx`.

> Sin `-mavx` (o `-march=native`) la compilación falla con errores `target specific option mismatch`. Usa también `-O2`: sin optimización la comparación de tiempos no es representativa.

## Ejemplo de salida

Los valores cambian en cada ejecución (los coeficientes son aleatorios) y los tiempos dependen de tu procesador:

```
Primeros 10 coeficientes:
  coef[0] = 0.085000
  ...
Resultado Horner escalar:  8618.606445
Resultado Horner con AVX:  8617.589844
Tiempo Horner escalar: 2.447648 segundos
Tiempo Horner con AVX: 0.315876 segundos
```

## Parámetros

Se modifican al inicio de `polinomio_horner.c`:

- `N`: cantidad de coeficientes (**debe ser múltiplo de 8**)
- `NUM_TRIALS`: repeticiones para medir el tiempo
- `X` (en `main`): punto donde se evalúa el polinomio

## Notas

- **`X = 1.0001f`:** un `float` llega hasta ~3.4×10³⁸. Con valores como `X = 1.1` y grado 10 000, el resultado desborda y sale `inf`.
- **Diferencia entre resultados:** el resultado escalar y el AVX pueden diferir en los últimos dígitos, porque el orden de las operaciones cambia y el redondeo de `float` se acumula en 10 000 pasos.
- **Memoria alineada:** la carga `_mm256_load_ps` exige memoria alineada a 32 bytes, por eso se reserva con `_mm_malloc` y se libera con `_mm_free`.
- **Variable `volatile`:** evita que el compilador elimine los ciclos de medición al optimizar.

## Estructura

```
.
├── polinomio_horner.c
├── README.md
└── .gitignore
```
