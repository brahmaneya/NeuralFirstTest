/*
 * DataExtractor.cpp
 *
 *  Created on: Dec 15, 2014
 *      Author: manas
 */

#include "DataExtractor.h"
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>

using namespace std;
#define f(i,n) for(int i = 0; i < n; i++)

DataEntry::DataEntry(vector<double> input, int output) {
	this->input = input;
	this->output = output - 'A'; // Outputs in file are chars from A to Z. Mapping to 0 to 25.
}

DataExtractor::DataExtractor(string inputFile, int numOutputs, double trainFraction) : trainIndex(0), epochs(0) {
	this->numOutputs = numOutputs;
	ifstream fin(inputFile);
	string line = "";
	while (getline (fin, line) != NULL) {
		char* cline = new char[line.size() + 1];
		strcpy(cline, line.c_str());
		int i = 0;
		char* t = strtok(cline, ",");
		char output = t[0];
		t = strtok(NULL, ",");
		vector<double> input;
		while (t != NULL) {
			input.push_back(atof(t));
			t = strtok(NULL, ",");
		}
		DataEntry newEntry(input, output);
		if (rand() % 100 < trainFraction * 100) {
			this->trainingTuples.push_back(newEntry);
		} else {
			this->testTuples.push_back(newEntry);
		}
	}
	fin.close();
	random_shuffle(trainingTuples.begin(), trainingTuples.end());
	random_shuffle(testTuples.begin(), testTuples.end());
}

vector<DataEntry> DataExtractor::generateBatch (int batchSize) {
	vector<DataEntry> batch;
	f(i, batchSize) {
		batch.push_back(trainingTuples[trainIndex]);
		trainIndex++;
		if (trainIndex >= trainingTuples.size()) {
			trainIndex = 0;
			epochs++;
		}
	}
	return batch;
}

void DataExtractor::reset () {
	trainIndex = 0;
	epochs = 0;
}

int DataExtractor::getEpochs () {
	return epochs;
}

DataExtractor::~DataExtractor() {
	// TODO Auto-generated destructor stub
}

