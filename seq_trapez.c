#include <stdio.h>
#include <math.h>
#include <time.h>


float f(float x) {
    return 4 / (1 + x * x);
}

float trapezoidal_rule(float a, float b, int n) {
    float h = (b - a) / n;
    float x[n+1];
    for (int i = 0; i <= n; i++) {
        x[i] = a + i*h;
    }
    float s = f(x[0]) + f(x[n]);
    for (int i = 1; i < n; i++) {
        s += 2 * f(x[i]);
    }
    return (h/2) * s;
}

int main() {
    float a = 0, b = 1;
    int n = 128;
    clock_t start, end;
    float cpu_time_used;

    start = clock();
    float result = trapezoidal_rule(a, b, n);
    printf("estimated: %f\n", result);
    end = clock();
    cpu_time_used = ((float) (end - start)) / CLOCKS_PER_SEC;
    printf("Time elapsed: %lf\n", cpu_time_used);
    
    return 0;
}

