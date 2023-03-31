#include <stdio.h>
#include <time.h>

float power(float x, int n) {
    float res = 1.0;
    for (int i = 0; i < n; i++) {
        res *= x;
    }
    return res;
}

int factorial(int n) {
    int res = 1;
    for (int i = 1; i <= n; i++) {
        res *= i;
    }
    return res;
}

int main()
{
    float x, sin_x = 0;
    int n, i, sign = 1;
    n = 15;
    printf("Podaj x w radianach: ");
    scanf("%f", &x);

    clock_t start, end;
    float cpu_time_used;

    start = clock();

    for (i = 1; i <= n; i++)
    {
        sin_x += sign * power(x, 2 * i - 1) / factorial(2 * i - 1);
        sign *= -1;
    }

    printf("sin(%lf) = %lf\n", x, sin_x);

    end = clock();
    cpu_time_used = ((float) (end - start)) / CLOCKS_PER_SEC;
    printf("Time elapsed: %lf\n", cpu_time_used);

    return 0;
}
