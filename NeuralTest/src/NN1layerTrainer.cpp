/*
 * NN1layerTrainer.cpp
 *
 *  Created on: Dec 15, 2014
 *      Author: manas
 */

#include "NN1layerTrainer.h"
#include <iostream>
#include <cmath>

using namespace std;

#define f(i,n) for (int i = 0; i < n; i++)

NN1layerTrainer::NN1layerTrainer(NN1layer* nn1, DataExtractor* de1) : batchSize(BATCH_SIZE), maxEpochs(MAX_EPOCHS), changeThreshold(CHANGE_THRESHOLD), learnRate(LEARN_RATE), nn(nn1), de(de1) {}

void NN1layerTrainer::setTrainingParams(int batchSize, double changeThreshold, int maxEpochs, double learnRate) {
	this->batchSize = batchSize;
	this->changeThreshold = changeThreshold;
	this->maxEpochs = maxEpochs;
	this->learnRate = learnRate;
}

inline double correctOutput (int i, int output) {
	return i == output ? 1.0 : 0.0;
}

void NN1layerTrainer::train() {
	double lastError = de->trainingTuples.size() * nn->oSize; // Should be infinity
	double error = 0.0;
	double changeFraction = changeThreshold + 1.0; // something larger than threshold.
	int numEpochs = 0;
	double ** dihw = new double*[nn->iSize + 1];
	f(i, nn->iSize + 1) {
		dihw[i] = new double[nn->hSize];
	}
	double ** dhow = new double*[nn->hSize + 1];
	f(j, nn->hSize + 1) {
		dhow[j] = new double[nn->oSize];
	}
	double * input = new double[nn->iSize + 1];
	double * hidden = new double[nn->hSize + 1];
	double * output = new double[nn->oSize];
	double * dOutput = new double[nn->oSize];
	de->reset();
	while (numEpochs < maxEpochs && changeFraction > changeThreshold) {
		vector<DataEntry> batch = de->generateBatch(batchSize);
		f(i, nn->iSize + 1) {
			f(j, nn->hSize) {
				dihw[i][j] = 0.0;
			}
		}
		f(j, nn->hSize + 1) {
			f(k, nn->oSize) {
				dhow[j][k] = 0.0;
			}
		}
		f(bi, batchSize) {
			DataEntry entry = batch[bi];
			f(i, nn->iSize) {
				input[i] = entry.input[i];
			}
			f(j, nn->hSize) {
				hidden[j] = 0.0;
			}
			f(k, nn->oSize) {
				output[k] = 0.0;
			}
			nn->forward(input, hidden, output);
			f (k, nn->oSize) {
				dOutput[k] = correctOutput(k, entry.output) - output[k];
			}
			error += -log(output[entry.output]);
			nn->backProp(input, hidden, output, dOutput, dihw, dhow, learnRate);
		}
		f(i, nn->iSize) {
			f(j, nn->hSize) {
				nn->ihw[i][j] -= dihw[i][j];
			}
		}
		f(j, nn->hSize) {
			f(k, nn->oSize) {
				nn->how[j][k] -= dhow[j][k];
			}
		}

		if (de->getEpochs() > numEpochs) {
			numEpochs++;
			changeFraction = abs((lastError - error) / error);
			cout << numEpochs << " " << error / de->trainingTuples.size() << endl;
			lastError = error;
			error = 0.0;
		}
	}

	delete[] input;
	delete[] hidden;
	delete[] output;
	delete[] dOutput;
	f(i, nn->iSize) {
		delete[] dihw[i];
	}
	delete[] dihw;
	f(j, nn->hSize) {
		delete[] dhow[j];
	}
	delete[] dhow;
}

double clamp (double output) {
	return output > 0.5 ? 1.0 : 0.0;
}

int maxOut (double output[], int size) {
	int moi = -1;
	int mo = 0.0;
	f(k, size) {
		if (output[k] > mo) {
			mo = output[k];
			moi = k;
		}
	}
	return moi;
}


double NN1layerTrainer::test() {
	double crossEntropyError = 0.0;
	double * input = new double[nn->iSize];
	double * hidden = new double[nn->hSize];
	double * output = new double[nn->oSize];
	de->reset();
	double total = 0;
	f(index, de->testTuples.size()) {
		DataEntry entry = de->testTuples[index];
		f(i, nn->iSize) {
			input[i] = entry.input[i];
		}
		f(j, nn->hSize) {
			hidden[j] = 0.0;
		}
		f(k, nn->oSize) {
			output[k] = 0.0;
		}
		nn->forward(input, hidden, output);
		crossEntropyError += -log(output[entry.output]);
	}
	delete[] input;
	delete[] hidden;
	delete[] output;
	cout <<	(crossEntropyError / de->testTuples.size()) << endl; 
	return crossEntropyError;
}

NN1layerTrainer::~NN1layerTrainer() {}
