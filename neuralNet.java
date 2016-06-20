import java.util.Random;
import java.util.Arrays;
//NOTE:The above imports are not for the ANN itself.

class FFANN { //Feed Forward ANN.
	private static double[] activationFunction(double neuronInputs[],double neuronWeights[], double neuronBiasValue) {
		double netInput = 0.f;
		double[] returnValues = new double[2];
		for(int currentInput = 0;currentInput<neuronInputs.length;currentInput++) {
			netInput += neuronInputs[currentInput] * neuronWeights[currentInput];
		}
		netInput += neuronBiasValue; //* 1;
		returnValues[0] = netInput;
		returnValues[1] = (1.0/(1+(Math.pow(Math.E,-netInput))));
		return returnValues;
	}
	private int structure[];
	private double netInputs[];
	private double neurons[]; //NOTE:The input neurons are not included in this array AND The double is just the output of that neuron after its last activation function.
	private double weights[]; //NOTE:The order of the weights are important.
	private double biasValues[];
	private boolean initializationState = true;
	public int[] getStructure() {
		return structure;
	}
	public double[] getWeights() {
		return weights;
	}
	public boolean setWeights(double newWeights[]) {
		if(newWeights.length == weights.length) {
			weights = newWeights;
			return true;
		} else {
			return false;
		}
	}
	public double[] forwardPass(double inputs[]) { //NOTE:Each input neuron has just one input.
		double networkOutput[] = new double[structure[structure.length-1]];
		if(inputs.length == structure[0]) {
			double currentNeuronInputs[];
			double currentNeuronWeights[];
			int currentNeuronInLayer;
			int layerOffset = 0;
			int weightOffset = 0;
			double[] activationFunctionOutput = new double[2];
			for(int currentLayerInStructure = 1;currentLayerInStructure<structure.length;currentLayerInStructure++) {
				if(currentLayerInStructure == 1) {
					//Get input values:
					currentNeuronInputs = inputs;
				} else {
					//Get input values:
					currentNeuronInputs = new double[structure[currentLayerInStructure-1]];
					for(currentNeuronInLayer = 0;currentNeuronInLayer<currentNeuronInputs.length;currentNeuronInLayer++) {
						currentNeuronInputs[currentNeuronInLayer] = neurons[(currentNeuronInLayer+structure[currentLayerInStructure-2])-structure[0]];
					}
				}
				layerOffset += structure[currentLayerInStructure-1];
				if(currentLayerInStructure > 1) {
					weightOffset += structure[currentLayerInStructure-2] * structure[currentLayerInStructure-1];
				}
				currentNeuronWeights = new double[structure[currentLayerInStructure-1]];
				for(currentNeuronInLayer = 0;currentNeuronInLayer<structure[currentLayerInStructure];currentNeuronInLayer++) {
					//Get weight values:
					for(int currentNeuronInLastLayer = 0;currentNeuronInLastLayer<structure[currentLayerInStructure-1];currentNeuronInLastLayer++) {
						currentNeuronWeights[currentNeuronInLastLayer] = weights[(currentNeuronInLayer+(currentNeuronInLastLayer*structure[currentLayerInStructure])+weightOffset)];
					}
					activationFunctionOutput = activationFunction(currentNeuronInputs,currentNeuronWeights,biasValues[currentLayerInStructure-1]);
					netInputs[(layerOffset+currentNeuronInLayer)-structure[0]] = activationFunctionOutput[0];
					neurons[(layerOffset+currentNeuronInLayer)-structure[0]] = activationFunctionOutput[1];
					if(currentLayerInStructure == (structure.length-1)) {
						networkOutput[currentNeuronInLayer] = neurons[(layerOffset+currentNeuronInLayer)-structure[0]];
					}
				}
			}
		}
		return networkOutput;
	}
	public double train(double inputs[], double correctOutputs[], double learningRate) {
		double newWeights[] = new double[weights.length];
		double neuronErrors[] = new double[neurons.length];
		double passOutput[] = new double[structure[structure.length-1]];
		double totalError = 0;
		if(inputs.length == structure[0] && correctOutputs.length == structure[structure.length-1]) {
			passOutput = forwardPass(inputs);
			//Calculate total error:
			for(int currentOutput = 0;currentOutput<correctOutputs.length;currentOutput++) {
				totalError += Math.pow((0.5*(correctOutputs[currentOutput]-passOutput[currentOutput])),2);
			}
			//Preform Reverse Pass:
			double relativeEffect = 0;
			int hiddenNeuronWeights = (weights.length-(structure[structure.length-1]*structure[structure.length-2]));
			int currentWeight = 0;
			int weightOffset = weights.length;
			int neuronOffset = neurons.length;
			int currentNeuron = 0;
			double currentDelta = 0;
			int currentLayer;
			int currentNeuronInLayer;
			int currentWeightForNeuron;
			int currentNeuronInAboveLayer;
			int currentNeuronOffsetInAboveLayer;
			int connectingWeight;
			int weightInput;
			double eTotal;
			for(currentLayer = (structure.length-1);currentLayer>=1;currentLayer--) {
				weightOffset-=(structure[currentLayer]*structure[currentLayer-1]);
				neuronOffset-=structure[currentLayer];
				for(currentNeuronInLayer = (structure[currentLayer]-1);currentNeuronInLayer>=0;currentNeuronInLayer--) {
					currentNeuron = neuronOffset + currentNeuronInLayer;
					if(currentLayer == (structure.length-1)) {
						//neuronErrors[currentNeuron] = ((-1*(correctOutputs[(neurons.length-1)-currentNeuron]-neurons[currentNeuron]))*(neurons[currentNeuron]*(1-neurons[currentNeuron])));
						neuronErrors[currentNeuron] = ((-1*(correctOutputs[currentNeuronInLayer]-neurons[currentNeuron]))*(neurons[currentNeuron]*(1-neurons[currentNeuron])));
					} else { 
						//neuronErrors[currentNeuron] = ((neurons)*())
					}
					for(currentWeightForNeuron = (structure[currentLayer-1]-1);currentWeightForNeuron>=0;currentWeightForNeuron--) {
						currentWeight = weightOffset + currentNeuronInLayer + (currentWeightForNeuron*structure[currentLayer]);
						if(currentLayer == (structure.length-1)) {
							//Output Layer
							newWeights[currentWeight] = weights[currentWeight]-(learningRate*(-1*(correctOutputs[currentNeuronInLayer]-neurons[currentNeuron])*(neurons[currentNeuron]*(1-neurons[currentNeuron]))*neurons[(structure[currentLayer-2]+currentWeightForNeuron)-structure[0]]));
						} else {
							//Hidden Layer
							currentDelta = 0;
							eTotal = 0;
							for(currentNeuronOffsetInAboveLayer = 0;currentNeuronOffsetInAboveLayer<structure[currentLayer+1];currentNeuronOffsetInAboveLayer++) {
								currentNeuronInAboveLayer = (currentNeuronOffsetInAboveLayer+(neurons.length-structure[currentLayer+1]));
								connectingWeight = ((weights.length-(structure[currentLayer+1]*structure[currentLayer]))+(currentNeuronOffsetInAboveLayer*structure[currentLayer]));
								eTotal+=(weights[connectingWeight]*neuronErrors[currentNeuronInAboveLayer]);
							}
							//if((currentLayer-1) < 0) {
								//neuronErrors[currentNeuron] = (neurons[currentNeuron]*(1-neurons[currentNeuron]))*(inputs[currentWeightForNeuron]);
							//} else {
								weightInput = currentWeightForNeuron+(currentNeuron-currentNeuronInLayer);
								neuronErrors[currentNeuron] = (neurons[currentNeuron]*(1-neurons[currentNeuron]))*(neurons[weightInput]);
							//}
							currentDelta = eTotal*neuronErrors[currentNeuron];
							newWeights[currentWeight] = weights[currentWeight]-(learningRate*currentDelta);
						}
					}
				}
			}
		}
		weights = newWeights;
		return totalError;
	}
	public boolean isInitializedCorrectly() {
		return initializationState; //This function exists to make sure that the inputs to the class constructer matched up and also because then the client cannot set the initializationState.
	}
	FFANN(int ANNStructure[],double ANNBiasValues[],double ANNInitialWeights[]) { 
		//ANNStructure is array like such: {4,5,5,5,2} for a neural network with 5 layers, and 4 neruons for the first input layer etc...
		//ANNBiasValues is the bias values for all of the hidden layers and output layer.
		//ANNInitialWeights is a double array that is just all of the wieghts in the neural network.
		if(ANNStructure.length >= 1 && ANNStructure[0] >= 1 && ANNBiasValues.length == (ANNStructure.length-1)) {
			structure = ANNStructure;
			int tempNeuronCount = 0;
			int tempWeightCount = 0;
			for(int currentLayer = 0;currentLayer<structure.length;currentLayer++) {
				//NOTE:Input neurons have no weights so there are no weights generated for the inputs.
				if(currentLayer > 0) {
					tempNeuronCount+=structure[currentLayer]; //NOTE:Input neron float values are not included in neuron array as they dont need to be.
					tempWeightCount += structure[currentLayer] * structure[currentLayer-1];
				}
			}
			weights = ANNInitialWeights;
			neurons = new double[tempNeuronCount];
			netInputs = new double[tempNeuronCount];
			biasValues = ANNBiasValues;
			if(ANNInitialWeights.length != tempWeightCount) {
				initializationState = false;
			}
		} else {
			initializationState = false;
		}
	}
}

class ANNTools {
	public static Random random = new Random();
	public static FFANN createFFANN(int structure[],double minWeight,double maxWeight) {
		maxWeight = maxWeight-minWeight;
		int weightCount = 0;
		for(int currentLayer = 0;currentLayer<structure.length;currentLayer++) {
			if(currentLayer>0) {
				weightCount+=(structure[currentLayer]*structure[currentLayer-1]);
			}
			
		}
		double initialWeights[] = new double[weightCount];
		for(int currentInitialWeight = 0;currentInitialWeight<initialWeights.length;currentInitialWeight++) {
			initialWeights[currentInitialWeight] = (double)((random.nextDouble() * maxWeight)+minWeight);
		}
		double biasValues[] = new double[structure.length-1];
		for(int currentBias = 0;currentBias<(biasValues.length);currentBias++) {
			biasValues[currentBias] = 1.f;
		}
		return new FFANN(structure,biasValues,initialWeights);
	}
	public static FFANN createFFANN(int structure[]) {
		int weightCount = 0;
		for(int currentLayer = 0;currentLayer<structure.length;currentLayer++) {
			if(currentLayer>0) {
				weightCount+=(structure[currentLayer]*structure[currentLayer-1]);
			}
			
		}
		double initialWeights[] = new double[weightCount];
		for(int currentInitialWeight = 0;currentInitialWeight<initialWeights.length;currentInitialWeight++) {
			initialWeights[currentInitialWeight] = (double)((random.nextDouble() * 1.0)+0.0);
		}
		double biasValues[] = new double[structure.length-1];
		for(int currentBias = 0;currentBias<(biasValues.length);currentBias++) {
			biasValues[currentBias] = 1.f;
		}
		return new FFANN(structure,biasValues,initialWeights);
	}
	public static FFANN createFFANN(int structure[],double initialWeights[]) {
		int weightCount = 0;
		for(int currentLayer = 0;currentLayer<structure.length;currentLayer++) {
			if(currentLayer>0) {
				weightCount+=(structure[currentLayer]*structure[currentLayer-1]);
			}
			
		}
		if(initialWeights.length != weightCount) {
			initialWeights = new double[weightCount];
			for(int currentInitialWeight = 0;currentInitialWeight<initialWeights.length;currentInitialWeight++) {
				initialWeights[currentInitialWeight] = (double)((random.nextDouble() * 1.0)+0.0);
			}
		}
		double biasValues[] = new double[structure.length-1];
		for(int currentBias = 0;currentBias<(biasValues.length);currentBias++) {
			biasValues[currentBias] = 1.f;
		}
		return new FFANN(structure,biasValues,initialWeights);
	}
	public static FFANN createFFANN(int structure[],double minWeight,double maxWeight, double biasValue) {
		maxWeight = maxWeight-minWeight;
		int weightCount = 0;
		for(int currentLayer = 0;currentLayer<structure.length;currentLayer++) {
			if(currentLayer>0) {
				weightCount+=(structure[currentLayer]*structure[currentLayer-1]);
			}
			
		}
		double initialWeights[] = new double[weightCount];
		for(int currentInitialWeight = 0;currentInitialWeight<initialWeights.length;currentInitialWeight++) {
			initialWeights[currentInitialWeight] = (double)((random.nextDouble() * maxWeight)+minWeight);
		}
		double biasValues[] = new double[structure.length-1];
		for(int currentBias = 0;currentBias<(biasValues.length);currentBias++) {
			biasValues[currentBias] = biasValue;
		}
		return new FFANN(structure,biasValues,initialWeights);
	}
	public static FFANN createFFANN(int structure[],double minWeight,double maxWeight, double biasValues[]) {
		maxWeight = maxWeight-minWeight;
		int weightCount = 0;
		for(int currentLayer = 0;currentLayer<structure.length;currentLayer++) {
			if(currentLayer>0) {
				weightCount+=(structure[currentLayer]*structure[currentLayer-1]);
			}
			
		}
		double initialWeights[] = new double[weightCount];
		for(int currentInitialWeight = 0;currentInitialWeight<initialWeights.length;currentInitialWeight++) {
			initialWeights[currentInitialWeight] = (double)((random.nextDouble() * maxWeight)+minWeight);
		}
		if(biasValues.length != (structure.length-1)) {
			biasValues = new double[structure.length-1];
			for(int currentBias = 0;currentBias<(biasValues.length);currentBias++) {
				biasValues[currentBias] = 1.0;
			}
		}
		return new FFANN(structure,biasValues,initialWeights);
	}
}
class learnMethod {
	public double[] inputs;
	public double[] outputs;
	learnMethod() {
		
	}
}
class learnThread implements Runnable {
	public static int threadCount;
	public FFANN threadFFANN;
	public double desiredError;
	public double improvementThreshold;
	public int underImprovementLimit;
	public static double averageNetworkError = 1;
	public double networkError;
	public double lastNetworkError = 1;
	public double networkImprovement;
	public learnMethod learnMethods[];
	public double learningConstant;
	public static double lastSuccesfulWeights[];
	public static double lowestError = 1;
	learnThread(learnMethod initialLearnMethods[],double initialDesiredError /* what error you are trying to reach */,double initialImprovementThreshold /* what the lowest improvement can be */,int initialUnderImprovementLimit /* how many times it cannot meet the improvmentThreshold */,double initialLearningConstant,int structure[],double minWeight,double maxWeight,double biasValues[]) {
		threadFFANN = ANNTools.createFFANN(structure,minWeight,maxWeight,biasValues);
		desiredError = initialDesiredError;
		improvementThreshold = initialImprovementThreshold;
		underImprovementLimit = initialUnderImprovementLimit;
		learnMethods = initialLearnMethods;
		learningConstant = initialLearningConstant;
		threadCount++;
	}
	learnThread(learnMethod initialLearnMethods[],double initialDesiredError /* what error you are trying to reach */,double initialImprovementThreshold /* what the lowest improvement can be */,int initialUnderImprovementLimit /* how many times it cannot meet the improvmentThreshold */,double initialLearningConstant,int structure[],double minWeight,double maxWeight,double biasValue) {
		threadFFANN = ANNTools.createFFANN(structure,minWeight,maxWeight,biasValue);
		desiredError = initialDesiredError;
		improvementThreshold = initialImprovementThreshold;
		underImprovementLimit = initialUnderImprovementLimit;
		learnMethods = initialLearnMethods;
		learningConstant = initialLearningConstant;
		threadCount++;
	}
	learnThread(learnMethod initialLearnMethods[],double initialDesiredError /* what error you are trying to reach */,double initialImprovementThreshold /* what the lowest improvement can be */,int initialUnderImprovementLimit /* how many times it cannot meet the improvmentThreshold */,double initialLearningConstant,int structure[],double minWeight,double maxWeight) {
		threadFFANN = ANNTools.createFFANN(structure,minWeight,maxWeight);
		desiredError = initialDesiredError;
		improvementThreshold = initialImprovementThreshold;
		underImprovementLimit = initialUnderImprovementLimit;
		learnMethods = initialLearnMethods;
		learningConstant = initialLearningConstant;
		threadCount++;
	}
	learnThread(learnMethod initialLearnMethods[],double initialDesiredError /* what error you are trying to reach */,double initialImprovementThreshold /* what the lowest improvement can be */,int initialUnderImprovementLimit /* how many times it cannot meet the improvmentThreshold */,double initialLearningConstant,int structure[], double initialWeights[]) {
		threadFFANN = ANNTools.createFFANN(structure,initialWeights);
		desiredError = initialDesiredError;
		improvementThreshold = initialImprovementThreshold;
		underImprovementLimit = initialUnderImprovementLimit;
		learnMethods = initialLearnMethods;
		learningConstant = initialLearningConstant;
		threadCount++;
	}
	learnThread(learnMethod initialLearnMethods[],double initialDesiredError /* what error you are trying to reach */,double initialImprovementThreshold /* what the lowest improvement can be */,int initialUnderImprovementLimit /* how many times it cannot meet the improvmentThreshold */,double initialLearningConstant,int structure[]) {
		threadFFANN = ANNTools.createFFANN(structure);
		desiredError = initialDesiredError;
		improvementThreshold = initialImprovementThreshold;
		underImprovementLimit = initialUnderImprovementLimit;
		learnMethods = initialLearnMethods;
		learningConstant = initialLearningConstant;
		threadCount++;
	}
	public void run() {
		networkError = 0;
		averageNetworkError+=networkError;
		averageNetworkError=averageNetworkError/threadCount;
		int currentLearnMethod = 0;
		while(networkError>desiredError && underImprovementLimit >= 0) {
			for(currentLearnMethod=0;currentLearnMethod<learnMethods.length;currentLearnMethod++) {
				networkError+=threadFFANN.train(learnMethods[currentLearnMethod].inputs,learnMethods[currentLearnMethod].outputs,learningConstant);
			}
			networkError=networkError/learnMethods.length;
			networkImprovement = networkError-lastNetworkError;
			lastNetworkError = networkError;
			if(networkImprovement<improvementThreshold) {
				underImprovementLimit--;
			}
		}
		if(underImprovementLimit > 0) {
			lastSuccesfulWeights = threadFFANN.getWeights();
			if(networkError<lowestError) {
				lowestError = networkError;
			}
		}
		threadCount--;
	}
}

class neuralNet {
	public static void main(String args[]) {
		Random random = new Random();
		int structure[] = {2,5,1}; //Overall Structure
		//CURRENTLY LEARNING NAND GATE
		learnMethod learningMethods[] = {new learnMethod(),new learnMethod(),new learnMethod(),new learnMethod()};
		learningMethods[0].inputs = new double[2];
		learningMethods[0].outputs = new double[1];
		learningMethods[0].inputs[0] = 0.f;
		learningMethods[0].inputs[1] = 0.f;
		learningMethods[0].outputs[0] = 1.f;
		learningMethods[1].inputs = new double[2];
		learningMethods[1].outputs = new double[1];
		learningMethods[1].inputs[0] = 0.f;
		learningMethods[1].inputs[1] = 1.f;
		learningMethods[1].outputs[0] = 1.f;
		learningMethods[2].inputs = new double[2];
		learningMethods[2].outputs = new double[1];
		learningMethods[2].inputs[0] = 1.f;
		learningMethods[2].inputs[1] = 0.f;
		learningMethods[2].outputs[0] = 1.f;
		learningMethods[3].inputs = new double[2];
		learningMethods[3].outputs = new double[1];
		learningMethods[3].inputs[0] = 1.f;
		learningMethods[3].inputs[1] = 1.f;
		learningMethods[3].outputs[0] = 0.f;
		int weightCount = 0;
		for(int currentLayer = 0;currentLayer<structure.length;currentLayer++) {
			if(currentLayer>0) {
				weightCount+=(structure[currentLayer]*structure[currentLayer-1]);
			}
			
		}
		learnThread.lastSuccesfulWeights = new double[weightCount];
		for(int currentInitialWeight = 0;currentInitialWeight<learnThread.lastSuccesfulWeights.length;currentInitialWeight++) {
			learnThread.lastSuccesfulWeights[currentInitialWeight] = (double)((random.nextDouble() * 0.7)+0.3);
		}
		int threadLimit = 8;
		double desiredTotalError = 0.01;
		double initialDesiredTotalError = 0.5;
		double desiredImprovement = 0.0000001; //At least this much
		int improvementLimit = 100; //Can miss this many times
		double learningConstant = 1.f;
		FFANN idealNetwork = ANNTools.createFFANN(structure);
		double updateRate = 1000; //Every second
		double curTime = System.currentTimeMillis();
		double lastTime = System.currentTimeMillis();
		while(learnThread.threadCount<threadLimit) {
			new learnThread(learningMethods,initialDesiredTotalError,desiredImprovement,improvementLimit,learningConstant,structure);
		}
		while(learnThread.averageNetworkError>desiredTotalError) {
			curTime = System.currentTimeMillis();
			if((curTime-lastTime)>=updateRate) {
				System.out.println("Lowest Error: " + learnThread.lowestError + " Average Error: " + learnThread.averageNetworkError);
				lastTime=curTime;
				//System.out.println(idealNetwork.isInitializedCorrectly());
				idealNetwork.setWeights(learnThread.lastSuccesfulWeights);
				double outputs[];
				String outStr = "";
				for(int currentMethod = 0;currentMethod<learningMethods.length;currentMethod++) {
					outputs = idealNetwork.forwardPass(learningMethods[currentMethod].inputs);
					for(int currentOut = 0;currentOut<outputs.length;currentOut++) {
						outStr=("IN: ");
						for(int currentInput = 0;currentInput<learningMethods[currentMethod].inputs.length;currentInput++) {
							outStr+=(learningMethods[currentMethod].inputs[currentInput]+" ");
						}
						outStr+=("OUT: " + outputs[currentOut]);
						System.out.println(outStr);
					}
				}
			}
			while(learnThread.threadCount<threadLimit) {
				new learnThread(learningMethods,learnThread.averageNetworkError,desiredImprovement,improvementLimit,learningConstant,structure);
			}
		}
		//OLD CODE:
		/*
		Random random = new Random();
		//Generate and output nerual network structure.
		//Structure:
		int mahStructure[] = {2,5,5,1};
		int mahWeightCount = 0;
		System.out.println("Neural Network Structure:");
		for(int currentMahLayer = 0;currentMahLayer<mahStructure.length;currentMahLayer++) {
			if(currentMahLayer>0) {
				mahWeightCount+=(mahStructure[currentMahLayer]*mahStructure[currentMahLayer-1]);
			}
			System.out.println(" Layer " + (currentMahLayer+1) + ":");
			if(currentMahLayer == 0) {
				System.out.println("  Type: Input");
				System.out.println("   Neuron Count: " + mahStructure[currentMahLayer]);
			} else if(currentMahLayer == (mahStructure.length-1)) {
				System.out.println("  Type: Output");
				System.out.println("   Neuron Count: " + mahStructure[currentMahLayer]);
			} else {
				System.out.println("  Type: Hidden");
				System.out.println("   Neuron Count: " + mahStructure[currentMahLayer]);
			}
		}
		//Weights:
		double mahInitialWeights[] = new double[mahWeightCount]; //EX:2 * 5 for first layer, 5 * 5 for second, 5 * 1 for last. {2,5,5,1}
		System.out.println(" Weights:");
		for(int currentInitialWeight = 0;currentInitialWeight<mahInitialWeights.length;currentInitialWeight++) {
			mahInitialWeights[currentInitialWeight] = (double)((random.nextDouble() * 0.5)+0.5);
			System.out.println("  w" + (currentInitialWeight+1) + ": " + mahInitialWeights[currentInitialWeight]);
		}
		//Bias Values:
		double mahBiasValues[] = {1.f,1.f,1.f};
		System.out.println(" Bias Values:");
		for(int currentBiasValue = 0;currentBiasValue<mahBiasValues.length;currentBiasValue++) {
			System.out.println("  Layer " + (currentBiasValue+1) + " bias value: " + mahBiasValues[currentBiasValue]);
		}
		//Create Neural Network.
		FFANN mahANN = new FFANN(mahStructure, mahBiasValues, mahInitialWeights);
		//Make sure the nerual network initialized without any errors.
		if(mahANN.isInitializedCorrectly()) {
			System.out.println("\nThe ANN initialized correctly!");
			double mahLearningConstant = 0.1;
			double annInputs[] = new double[2];
			double annDesiredOutputs[] = new double[1];
			double mahTotalError = 1;
			long mahTrains = 0;
			long currentTime = System.currentTimeMillis();
			long lastTime = System.currentTimeMillis();
			long startTime = currentTime;
			double lastError = 1;
			double requiredError = 0.01;
			System.out.println("Starting training for XOR gate...");
			while(mahTotalError>requiredError) {
				mahTrains++;
				mahTotalError = 0;
				annInputs[0] = 0.f;
				annInputs[1] = 0.f;
				annDesiredOutputs[0] = 0.f;
				mahTotalError+=mahANN.train(annInputs,annDesiredOutputs,mahLearningConstant);
				annInputs[0] = 1.f;
				annInputs[1] = 0.f;
				annDesiredOutputs[0] = 1.f;
				mahTotalError+=mahANN.train(annInputs,annDesiredOutputs,mahLearningConstant);
				annInputs[0] = 0.f;
				annInputs[1] = 1.f;
				annDesiredOutputs[0] = 1.f;
				mahTotalError+=mahANN.train(annInputs,annDesiredOutputs,mahLearningConstant);
				annInputs[0] = 1.f;
				annInputs[1] = 1.f;
				annDesiredOutputs[0] = 0.f;
				mahTotalError+=mahANN.train(annInputs,annDesiredOutputs,mahLearningConstant);
				mahTotalError=mahTotalError/4;
				currentTime = System.currentTimeMillis();
				if((currentTime-lastTime)>=1000) {
					lastTime = currentTime;
					System.out.println("Generation " + mahTrains + ":");
					System.out.println(" Total Error:" + mahTotalError);
					System.out.println(" New Weights:");
					for(int mahCurWeight = 0;mahCurWeight<mahANN.getWeights().length;mahCurWeight++) {
						System.out.println("  w"+mahCurWeight+": "+mahANN.getWeights()[mahCurWeight]);
					}
					System.out.println(" Improvement:" + (lastError-mahTotalError));
					lastError=mahTotalError;
				}
			}
			System.out.println("Generation " + mahTrains + ":");
			System.out.println(" Total Error:" + mahTotalError);
			System.out.println(" New Weights:");
			for(int mahCurWeight = 0;mahCurWeight<mahANN.getWeights().length;mahCurWeight++) {
				System.out.println("  w"+mahCurWeight+": "+mahANN.getWeights()[mahCurWeight]);
			}
			System.out.println(" Improvement: " + (lastError-mahTotalError));
			long totalElapsedTime = (System.currentTimeMillis()-startTime);
			System.out.println("Total elapsed time: " + totalElapsedTime + " milliseconds.");
			double output[];
			double inputs[] = {0.f,0.f};
			output = mahANN.forwardPass(inputs);
			System.out.println("In: 0,0 Out: " + Math.round(output[0]));
			inputs[0] = 0.f;
			inputs[1] = 1.f;
			output = mahANN.forwardPass(inputs);
			System.out.println("In: 0,1 Out: " + Math.round(output[0]));
			inputs[0] = 1.f;
			inputs[1] = 0.f;
			output = mahANN.forwardPass(inputs);
			System.out.println("In: 1,0 Out: " + Math.round(output[0]));
			inputs[0] = 1.f;
			inputs[1] = 1.f;
			output = mahANN.forwardPass(inputs);
			System.out.println("In: 1,1 Out: " + Math.round(output[0]));
		} else {
			mahANN = null;
			System.gc(); //Call the garbage collector.
			System.out.println("The ANN did not initialize correctly... Make sure to check that there are enough nodes to support an ANN and that there is a bias set for each hidden and output layer.");
		} 
		*/
	}
}
