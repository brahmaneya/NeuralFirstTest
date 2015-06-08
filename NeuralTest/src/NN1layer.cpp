/*
 * NN1layer.cpp
 *
 *  Created on: Dec 14, 2014
 *      Author: manas
 */

#include "NN1layer.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <fstream>

#define f(i,n) for (int i = 0; i < n; i++)

using namespace std;

NN1layer::NN1layer(int iSize, int hSize, int oSize) {
	this->iSize = iSize;
	this->hSize = hSize;
	this->oSize = oSize;

	ihw = new double*[iSize + 1];
	f(i, iSize + 1) {
		ihw[i] = new double[hSize];
	}
	how = new double*[hSize + 1];
	f(j, hSize + 1) {
		how[j] = new double[oSize];
	}


	initWeights();
}

NN1layer::NN1layer(string fileName) : iSize(0), hSize(0), oSize(0) {
	ifstream fin(fileName);
	fin >> iSize >> hSize >> oSize;
	ihw = new double*[iSize + 1];
	f(i, iSize + 1) {
		ihw[i] = new double[hSize];
	}
	how = new double*[hSize + 1];
	f(j, hSize + 1) {
		how[j] = new double[oSize];
	}
	f(i, iSize + 1) {
		f(j, hSize) {
			fin >> ihw[i][j];
		}
	}
	f(j, hSize + 1) {
		f(k, oSize) {
			fin >> how[j][k];
		}
	}
	fin.close();
}

void NN1layer::initWeights() {
	double varH = 1 / sqrt( (double) hSize);
	double varO = 1 / sqrt( (double) oSize);
	srand (time(NULL));
	f(i, iSize + 1) {
		f(j, hSize) {
			ihw[i][j] = varH * (rand() % 100 - 50.0) / 50;
		}
	}
	f(j, hSize + 1) {
		f(k, oSize) {
			how[j][k] = varO * (rand() % 100 - 50.0) / 50;
		}
	}
}

void NN1layer::forward(double input[], double hidden[], double output[]) {
	input[iSize] = 1;
	f(j, hSize) {
		hidden[j] = 0.0;
		f(i, iSize) {
			hidden[j] += input[i] * ihw[i][j];
		}
		hidden[j] += ihw[iSize][j]; // Bias term
		hidden[j] = activationFunction(hidden[j]);
	}
	hidden[hSize] = 1;
	f(k, oSize) {
		output[k] = 0.0;
		f(j, hSize) {
			output[k] += hidden[j] * how[j][k];
		}
		output[k] += how[hSize][k]; // Bias term
	}
	softMaxFunction(output, oSize);
}

/**
 * dhwi and dhow have to be pre-initialized, and set to zero if this is the first member of the batch. The backprop will simply add to them.
 */
void NN1layer::backProp(double input[], double hidden[], double output[], double dOutput[], double ** dihw, double ** dhow, double learnRate) {
	double* dOut = new double[oSize];
	f(k, oSize) {
		dOut[k] = -dOutput[k];
	}
	double* dHidden = new double[hSize];
	double* dHid = new double[hSize];
	f(j, hSize) {
		dHidden[j] = 0;
		f(k, oSize) {
			dHidden[j] += dOut[k] * how[j][k];
		}
		dHid[j] = dHidden[j] * activationDerivativeFunction(hidden[j]);
	}

	f(j, hSize + 1) {
		f(k, oSize) {
			dhow[j][k] += dOut[k] * learnRate * hidden[j];
		}
	}

	f(i, iSize + 1) {
		f(j, hSize) {
			dihw[i][j] += dHid[j] * learnRate * input[i];
		}
	}
	delete[] dOut;
	delete[] dHidden;
	delete[] dHid;
}

void NN1layer::softMaxFunction(double output[], int size) {
	double normalize = 0.0;
	f(k, size) {
		normalize += exp(output[k]);
	}
	f(k, size) {
		output[k] = exp(output[k]) / normalize;
	}
	return;
}

inline double NN1layer::activationFunction(double x) {
	return 1 / (1 + exp(-x));
}

inline double NN1layer::activationDerivativeFunction(double x) {
	return x * (1 - x);
}

void NN1layer::save (string fileName) {
	ofstream fout(fileName);
	fout << iSize << " " << hSize << " " << oSize << endl;
	f(i, iSize + 1) {
		f(j, hSize) {
			fout << ihw[i][j] << " ";
		}
	}
	f(j, hSize + 1) {
		f(k, oSize) {
			fout << how[j][k] << " ";
		}
	}
	fout << endl;
	fout.close();
}

void NN1layer::load (string fileName) {
	ifstream fin(fileName);
	fin >> iSize >> hSize >> oSize;
	ihw = new double*[iSize + 1];
	f(i, iSize + 1) {
		ihw[i] = new double[hSize];
	}
	how = new double*[hSize + 1];
	f(j, hSize + 1) {
		how[j] = new double[oSize];
	}
	f(i, iSize + 1) {
		f(j, hSize) {
			fin >> ihw[i][j];
		}
	}
	f(j, hSize + 1) {
		f(k, oSize) {
			fin >> how[j][k];
		}
	}
	fin.close();
}

NN1layer::~NN1layer() {
	f(i, iSize) {
		delete[] ihw[i];
	}
	delete[] ihw;

	f(j, hSize) {
		delete[] how[j];
	}
	delete[] how;
}

