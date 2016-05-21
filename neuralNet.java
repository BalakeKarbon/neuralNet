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
			for(currentLayer = (structure.length-1);currentLayer>=1;currentLayer--) {
				weightOffset-=(structure[currentLayer]*structure[currentLayer-1]);
				neuronOffset-=structure[currentLayer];
				for(currentNeuronInLayer = (structure[currentLayer]-1);currentNeuronInLayer>=0;currentNeuronInLayer--) {
					currentNeuron = neuronOffset + currentNeuronInLayer;
					for(currentWeightForNeuron = (structure[currentLayer-1]-1);currentWeightForNeuron>=0;currentWeightForNeuron--) {
						currentWeight = weightOffset + currentNeuronInLayer + (currentWeightForNeuron*structure[currentLayer]);
						if(currentLayer == (structure.length-1)) {
							//Output Layer
							newWeights[currentWeight] = weights[currentWeight]-(learningRate*(-1*(correctOutputs[currentNeuronInLayer]-neurons[currentNeuron])*(neurons[currentNeuron]*(1-neurons[currentNeuron]))*neurons[(structure[currentLayer-2]+currentWeightForNeuron)-structure[0]]));
						} else {
							//Hidden Layer
							currentDelta = 0;
							for(currentNeuronInAboveLayer = 0;currentNeuronInAboveLayer<structure[currentLayer+1];currentNeuronInAboveLayer++) {
								//currentDelta+=();//HERE!!!!
							}
						}
					}
				}
			}
			//TO-DO: Back-Propagation continued!
		}
		//NOTE: Return new weights. Maybe make a new function to return total error.
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

class neuralNet {
	public static void main(String args[]) {
		Random random = new Random();
		//Generate and output nerual network structure.
		//Structure:
		int mahStructure[] = {2,5,1};
		System.out.println("Neural Network Structure:");
		for(int currentMahLayer = 0;currentMahLayer<mahStructure.length;currentMahLayer++) {
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
		double mahInitialWeights[] = new double[15]; //EX:2 * 5 for first layer, 5 * 5 for second, 5 * 1 for last. {2,5,5,1}
		System.out.println(" Weights:");
		for(int currentInitialWeight = 0;currentInitialWeight<mahInitialWeights.length;currentInitialWeight++) {
			mahInitialWeights[currentInitialWeight] = (double)((random.nextDouble() * 0.7)+0.3);
			System.out.println("  w" + (currentInitialWeight+1) + ": " + mahInitialWeights[currentInitialWeight]);
		}
		//Bias Values:
		double mahBiasValues[] = {1.f,1.f};
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
			for(int mahCurrentPass = 0;mahCurrentPass<1;mahCurrentPass++) {
				annInputs[0] = 0.f;
				annInputs[1] = 0.f;
				annDesiredOutputs[0] = 1.f;
				mahANN.train(annInputs,annDesiredOutputs,mahLearningConstant);
				annInputs[0] = 1.f;
				annInputs[1] = 0.f;
				annDesiredOutputs[0] = 1.f;
				mahANN.train(annInputs,annDesiredOutputs,mahLearningConstant);
				annInputs[0] = 0.f;
				annInputs[1] = 1.f;
				annDesiredOutputs[0] = 1.f;
				mahANN.train(annInputs,annDesiredOutputs,mahLearningConstant);
				annInputs[0] = 1.f;
				annInputs[1] = 1.f;
				annDesiredOutputs[0] = 0.f;
				mahANN.train(annInputs,annDesiredOutputs,mahLearningConstant);
			}
		} else {
			mahANN = null;
			System.gc(); //Call the garbage collector.
			System.out.println("The ANN did not initialize correctly... Make sure to check that there are enough nodes to support an ANN and that there is a bias set for each hidden and output layer.");
		}
	}
}
