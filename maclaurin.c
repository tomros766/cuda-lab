#include <stdio.h>
#include <math.h>

int main()
{
    double x, sin_x = 0;
    int n, i, sign = 1;

    printf("Podaj x w radianach: ");
    scanf("%lf", &x);

    printf("Podaj dokładność N: ");
    scanf("%d", &n);

    for (i = 1; i <= n; i++)
    {
        sin_x += sign * pow(x, 2 * i - 1) / tgamma(2 * i);
        sign *= -1;
    }

    printf("sin(%lf) = %lf", x, sin_x);

    return 0;
}

