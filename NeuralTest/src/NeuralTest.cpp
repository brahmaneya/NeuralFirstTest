//============================================================================
// Name        : NeuralTest.cpp
// Author      : Manas Joglekar
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <string>
#include "DataExtractor.h"
#include "NN1layer.h"
#include "NN1layerTrainer.h"

using namespace std;

int main() {
	double lambda = 0.000001;
	int batchSize = 10;
	int maxEpochs = 100;
	double changeThreshold = -0.001;
	double trainFraction = 0.8;
	double learnRate = 0.002;
	int iSize = 16;
	int hSize = 20;
	int oSize = 26;
	string dataFile = "data/letter-recognition.data";
	string saveFile = "nn_16_20_26";
	DataExtractor *de = new DataExtractor(dataFile, oSize, trainFraction);
	NN1layer *nn = new NN1layer(16, 20, 26);
	//NN1layer *nn = new NN1layer(saveFile);
	NN1layerTrainer nnTrainer(nn, de);
	nnTrainer.setTrainingParams(batchSize, changeThreshold, maxEpochs, lambda, learnRate);
	nnTrainer.train();
	nn->save(saveFile);
	nnTrainer.test();
	return 0;
}
