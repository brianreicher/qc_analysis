#include <iostream>
#include <math.h>

/*
y=[]
for i in range(1,1001):
    I=0.
    n0=i
    N=n0
    r=np.random.random(n0)
    for i in range(n0):
        x=r[i]
        I+=np.sqrt(1-x**2)
    I*=4./float(n0)
    y.append(I)
x=np.arange(1,1001,1)
 */


float simpleMonteCarlo(int range,  float i = 0.)
{
    float n_o = i;
    int randArr[range];
    for (int j = 0; j < range; j++) {
        randArr[j] = rand() % 1;
    }
    for (int k = 0; k < range; k++)
    {
        int x = randArr[k];
        i += sqrt(1- pow(x, 2));
    }
    i *= 4./n_o;

    return i;
}


int main()
{
    int range;
    std::cout << "Enter a Monte Carlo iteration range: ";
    std::cin >> range;
    std::cout << "Monte Carlo Estimation: " << simpleMonteCarlo(range);
    return 0;
}