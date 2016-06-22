//All of this code was just put in the main method and executed. No classes are needed besides FFANN.

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
