#include <stdio.h>
#include <math.h>

double f(double x) {
    return 4 / (1 + x * x);
}

double trapezoidal_rule(double a, double b, int n) {
    double h = (b - a) / n;
    double x[n+1];
    for (int i = 0; i <= n; i++) {
        x[i] = a + i*h;
    }
    double s = f(x[0]) + f(x[n]);
    for (int i = 1; i < n; i++) {
        s += 2 * f(x[i]);
    }
    return (h/2) * s;
}

int main() {
    double a = 0, b = 1;
    int n = 128;
    double result = trapezoidal_rule(a, b, n);
    printf("estimated: %f\n", result);
    return 0;
}

