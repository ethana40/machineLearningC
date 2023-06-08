#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* This is a very simple single neron machine learning model.  It is not very good.
 This was primarily used as practice and familiarization with how machine learning works in general.
 It attempts to guess what number you need to multiply the first number by to get the second number*/
 
// "training data"
float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
};
#define trainCount (sizeof(train)/sizeof(train[0]))

float randFloat(void)
{
    return (float) rand()/ (float) RAND_MAX;
}

float cost(float w, float b)
{
    float result = 0.0f;
    for (size_t i = 0; i < trainCount; i++){
        float x = train[i][0];
        float y = x*w + b;
        float d = y - train[i][1];
        result += d*d;
    }
    result /= trainCount;
    return result;
}

int main()
{
    // creating random weight and bias
    srand(time(0));
    float w = randFloat() * 10.0f;
    float b = randFloat() * 5.0f;

    // epsilon and learning rate
    float eps = 1e-4;
    float rate = 1e-4;

    // adjusts weights and biases by approximating the derivative of the weight and bias
    for (size_t i = 0; i < 1500; i++){
        float c = cost(w, b);
        float dw = (cost(w + eps, b) - c)/eps;
        float db = (cost(w, b + eps) - c)/eps;
        w -= rate*dw;
        b -= rate*dw;
        printf("cost = %f, w = %f, b = %f\n", cost(w, b), w, b);
    }
    
    printf("w = %f b = %f\nOutput: %f\n", w, b, w+b);

    for (size_t x = 0; x < trainCount; x++)
    {
        float actual = (w + b) * train[x][0];
        printf("Expected: %f Actual %f\n", train[x][1], actual);
    }

    return 0;
}