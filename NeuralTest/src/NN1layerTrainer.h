/*
 * NN1layerTrainer.h
 *
 *  Created on: Dec 15, 2014
 *      Author: manas
 */
#include "NN1layer.h"
#include "DataExtractor.h"

#ifndef NN1LAYERTRAINER_H_
#define NN1LAYERTRAINER_H_

#define MAX_EPOCHS 2000
#define LEARN_RATE 0.001
#define CHANGE_THRESHOLD 0.01
#define BATCH_SIZE 10

class NN1layerTrainer {
private:
	int batchSize;
	double changeThreshold;
	int maxEpochs;
	NN1layer* nn;
	DataExtractor* de;
	double learnRate;

public:
	NN1layerTrainer(NN1layer* nn1, DataExtractor* de1);
	void setTrainingParams(int batchSize, double changeThreshold, int maxEpochs, double learnRate);
	void train();
	double test();
	virtual ~NN1layerTrainer();
};

#endif /* NN1LAYERTRAINER_H_ */
