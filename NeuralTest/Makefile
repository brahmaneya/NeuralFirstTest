CXXFLAGS =	-O2 -g -Wall -fmessage-length=0

OBJS =		NeuralTest.o DataExtractor.o NN1layer.o NN1layerTrainer.o 

LIBS =

TARGET =	neuraltest

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

NeuralTest.o:	src/NeuralTest.cpp
	$(CXX) -c src/NeuralTest.cpp

DataExtractor.o:	src/DataExtractor.cpp
	$(CXX) -c src/DataExtractor.cpp

NN1layer.o:	src/NN1layer.cpp
	$(CXX) -c src/NN1layer.cpp

NN1layerTrainer.o:	src/NN1layerTrainer.cpp
	$(CXX) -c src/NN1layerTrainer.cpp

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
