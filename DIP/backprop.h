#pragma once

struct NN {
	int* n; // respects the neuron
	int l; // number of layers
	double*** w; // scales

	double* in; // input vector
	double* out; // output the vector
	double** y; //output layer vectors

	double** d; // neuron errors
};

NN* createNN(int n, int h, int o);
void releaseNN(NN*& nn);
void feedforward(NN* nn);
double backpropagation(NN* nn, double* t);
void setInput(NN* nn, double* in, bool verbose = false);
int getOutput(NN* nn, bool verbose = false);