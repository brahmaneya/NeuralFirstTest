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

NN1layerTrainer::NN1layerTrainer(NN1layer* nn1, DataExtractor* de1) : batchSize(BATCH_SIZE), maxEpochs(MAX_EPOCHS), changeThreshold(CHANGE_THRESHOLD), lambda(LAMBDA), learnRate(LEARN_RATE), nn(nn1), de(de1) {}

void NN1layerTrainer::setTrainingParams(int batchSize, double changeThreshold, int maxEpochs, double lambda, double learnRate) {
	this->batchSize = batchSize;
	this->changeThreshold = changeThreshold;
	this->maxEpochs = maxEpochs;
	this->lambda = lambda;
	this->learnRate = learnRate;
}

inline double correctOutput (int i, int output) {
	return i == output ? 1.0 : 0.0;
}

double regularize(double lambda, double ** ihw, double ** how, double ** dihw, double ** dhow, int iSize, int hSize, int oSize) {
	double error = 0;
	f(i, iSize) {
		f(j, hSize) {
			dihw[i][j] += ihw[i][j] * lambda;
			error += ihw[i][j] * ihw[i][j];
		}
	}
	f(j, hSize) {
		f(k, oSize) {
			dhow[j][k] += how[j][k] * lambda;
			error += how[j][k] * how[j][k];
		}
	}
	return 0.5 * lambda * error;
}

void NN1layerTrainer::train() {
	double lastError = de->trainingTuples.size() * nn->oSize; // Should be infinity
	double error = 0.0;
	double changeFraction = changeThreshold + 1.0; // something larger than threshold.
	double rms = 0.0; // for RMSProp.
	double rmsWindowFactor = 0.9;
	double momentum = 0.1; // momentum parameter
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
				dOutput[k] = output[k] - correctOutput(k, entry.output);
			}
			error += -log(output[entry.output]);
			nn->backProp(input, hidden, output, dOutput, dihw, dhow, learnRate);
		}
		// Regurilzation commented out because its not improving test case performance.
		//error += regularize(lambda, nn->ihw, nn->how, dihw, dhow, nn->iSize, nn->hSize, nn->oSize);

		// rmsprop part
		rms *= rmsWindowFactor;
		f(i, nn->iSize + 1) {
			f(j, nn->hSize) {
				rms += (1 - rmsWindowFactor) * dihw[i][j] * dihw[i][j];
			}
		}
		f(j, nn->hSize + 1) {
			f(k, nn->oSize) {
				rms += (1 - rmsWindowFactor) * dhow[j][k] * dhow[j][k];
			}
		}

		f(i, nn->iSize + 1) {
			f(j, nn->hSize) {
				nn->ihw[i][j] -= learnRate * dihw[i][j] / ((1 - rmsWindowFactor) * sqrt(rms));
			}
		}
		f(j, nn->hSize + 1) {
			f(k, nn->oSize) {
				nn->how[j][k] -= learnRate * dhow[j][k] / ((1 - rmsWindowFactor) * sqrt(rms));
			}
		}

		/* 
		// non-rmsprop
		f(i, nn->iSize + 1) {
			f(j, nn->hSize) {
				nn->ihw[i][j] -= dihw[i][j];
			}
		}
		f(j, nn->hSize + 1) {
			f(k, nn->oSize) {
				nn->how[j][k] -= dhow[j][k];
			}
		}

		*/

		if (de->getEpochs() > numEpochs) {
			numEpochs++;
			changeFraction = abs((lastError - error) / error);
			cout << numEpochs << " " << error / de->trainingTuples.size() << endl;
			if (lastError < error) {
				learnRate /= 2;
			}
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
	double mo = 0.0;
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
	double errorClassification = 0.0;
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
		errorClassification += abs(1 - correctOutput(maxOut(output, nn->oSize), entry.output));
		crossEntropyError += -log(output[entry.output]);
	}
	delete[] input;
	delete[] hidden;
	delete[] output;
	cout <<	(errorClassification / de->testTuples.size()) << "\t" <<
			(crossEntropyError / de->testTuples.size()) << endl;
	return errorClassification;
}

NN1layerTrainer::~NN1layerTrainer() {}
