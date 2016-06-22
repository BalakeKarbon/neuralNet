import java.io.*;
import java.io.File.*;
import java.util.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.event.*;
import java.awt.*;
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
	public double[] getBiasValues() {
		return biasValues;
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
	public static int threadLimit = 1;
	public FFANN threadFFANN;
	public double desiredError;
	public double improvementThreshold;
	public int underImprovementLimit;
	public static double averageNetworkError = 1;
	public static double averageNetworkImprovement = 0;
	public double networkError = 1;
	public double lastNetworkError = 1;
	public double networkImprovement;
	public learnMethod learnMethods[];
	public double learningConstant;
	public static double lastSuccesfulWeights[];
	public static double lowestError = 1;
	public static int recycledThreads = 0;
	public static int totalGenerations = 0;
	public int threadGens = 0;
	learnThread(learnMethod initialLearnMethods[],double initialDesiredError /* what error you are trying to reach */,double initialImprovementThreshold /* what the lowest improvement can be */,int initialUnderImprovementLimit /* how many times it cannot meet the improvmentThreshold */,double initialLearningConstant,int structure[],double minWeight,double maxWeight,double biasValues[]) {
		if(threadCount<(threadLimit+1)) {
			threadFFANN = ANNTools.createFFANN(structure,minWeight,maxWeight,biasValues);
			desiredError = initialDesiredError;
			improvementThreshold = initialImprovementThreshold;
			underImprovementLimit = initialUnderImprovementLimit;
			learnMethods = initialLearnMethods;
			learningConstant = initialLearningConstant;
			(new Thread(this, ("neural network"))).start();
			threadCount++;
		}
	}
	learnThread(learnMethod initialLearnMethods[],double initialDesiredError /* what error you are trying to reach */,double initialImprovementThreshold /* what the lowest improvement can be */,int initialUnderImprovementLimit /* how many times it cannot meet the improvmentThreshold */,double initialLearningConstant,int structure[],double minWeight,double maxWeight,double biasValue) {
		if(threadCount<(threadLimit+1)) {
			threadFFANN = ANNTools.createFFANN(structure,minWeight,maxWeight,biasValue);
			desiredError = initialDesiredError;
			improvementThreshold = initialImprovementThreshold;
			underImprovementLimit = initialUnderImprovementLimit;
			learnMethods = initialLearnMethods;
			learningConstant = initialLearningConstant;
			(new Thread(this, ("neural network"))).start();
			threadCount++;
		}
	}
	learnThread(learnMethod initialLearnMethods[],double initialDesiredError /* what error you are trying to reach */,double initialImprovementThreshold /* what the lowest improvement can be */,int initialUnderImprovementLimit /* how many times it cannot meet the improvmentThreshold */,double initialLearningConstant,int structure[],double minWeight,double maxWeight) {
		if(threadCount<(threadLimit+1)) {
			threadFFANN = ANNTools.createFFANN(structure,minWeight,maxWeight);
			desiredError = initialDesiredError;
			improvementThreshold = initialImprovementThreshold;
			underImprovementLimit = initialUnderImprovementLimit;
			learnMethods = initialLearnMethods;
			learningConstant = initialLearningConstant;
			(new Thread(this, ("neural network"))).start();
			threadCount++;
		}
	}
	learnThread(learnMethod initialLearnMethods[],double initialDesiredError /* what error you are trying to reach */,double initialImprovementThreshold /* what the lowest improvement can be */,int initialUnderImprovementLimit /* how many times it cannot meet the improvmentThreshold */,double initialLearningConstant,int structure[], double initialWeights[]) {
		if(threadCount<(threadLimit+1)) {
			threadFFANN = ANNTools.createFFANN(structure,initialWeights);
			desiredError = initialDesiredError;
			improvementThreshold = initialImprovementThreshold;
			underImprovementLimit = initialUnderImprovementLimit;
			learnMethods = initialLearnMethods;
			learningConstant = initialLearningConstant;
			(new Thread(this, ("neural network"))).start();
			threadCount++;
		}
	}
	learnThread(learnMethod initialLearnMethods[],double initialDesiredError /* what error you are trying to reach */,double initialImprovementThreshold /* what the lowest improvement can be */,int initialUnderImprovementLimit /* how many times it cannot meet the improvmentThreshold */,double initialLearningConstant,int structure[]) {
		if(threadCount<(threadLimit+1)) {
			threadFFANN = ANNTools.createFFANN(structure);
			desiredError = initialDesiredError;
			improvementThreshold = initialImprovementThreshold;
			underImprovementLimit = initialUnderImprovementLimit;
			learnMethods = initialLearnMethods;
			learningConstant = initialLearningConstant;
			(new Thread(this, ("neural network"))).start();
			threadCount++;
		}
	}
	public void run() {
		int currentLearnMethod = 0;
		while(networkError>desiredError && underImprovementLimit >= 0) {
			networkError = 0;
			for(currentLearnMethod=0;currentLearnMethod<learnMethods.length;currentLearnMethod++) {
				networkError+=threadFFANN.train(learnMethods[currentLearnMethod].inputs,learnMethods[currentLearnMethod].outputs,learningConstant);
				threadGens++;
			}
			networkError=networkError/learnMethods.length;
			networkImprovement = networkError-lastNetworkError;
			lastNetworkError = networkError;
			//System.out.println(networkImprovement);
			if(networkImprovement<improvementThreshold) {
				underImprovementLimit--;
				//System.out.println("miss");
			}
			averageNetworkError+=networkError;
			averageNetworkError=averageNetworkError/threadLimit;
			averageNetworkImprovement+=networkImprovement;
			averageNetworkImprovement=averageNetworkImprovement/threadLimit;
		}
		//System.out.println("dead.");
		if(networkError<lowestError) {
			lowestError = networkError;
			lastSuccesfulWeights = threadFFANN.getWeights();
		}
		threadCount--;
		recycledThreads++;
		totalGenerations+=threadGens;
	}
}
class paintPanel extends JPanel implements ComponentListener,MouseListener,MouseMotionListener {
	public int width,height;
	public int pixelWidth,pixelHeight;
	public boolean drawing = false;
	public int[][] pixelArray;
	public int arrayX = 0;
	public int arrayY = 0;
	public int lastX = arrayX;
	public int lastY = arrayY;
	public boolean getDrawing = false;
	public double timeOfRelease = System.currentTimeMillis();
	public boolean isReady = false;
	public int resX = 0;
	public int resY = 0;
	public int curX,curY;
	public int curX2,curY2;
	public int missingPointsX,missingPointsY,missingPoints;
	public int curMPoint;
	public int tmpx = 0;
	public int tmpy = 0;
	public int tmpSlp=0;
	public boolean RMB = false;
	paintPanel(int rx,int ry) {
		resX = rx;
		resY = ry;
		width = getSize().width;
		height = getSize().height;
		pixelArray = new int[resX][resY];
		addComponentListener(this);
		pixelWidth = getSize().width/resX;
		pixelHeight = getSize().height/resY;
		addMouseListener(this);
		addMouseMotionListener(this);
	}
	@Override
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
        isReady = true;
		g.clearRect(0,0,width,height);
		for(curX = 0;curX<pixelArray.length;curX++) {
			for(curY = 0;curY<pixelArray[curX].length;curY++) {
				if(pixelArray[curX][curY] == 1) {
					g.setColor(Color.black);
				} else {
					g.setColor(Color.white);
				}
				g.fillRect((curX*pixelWidth),(curY*pixelHeight),(pixelWidth),(pixelHeight));
			}
		}
	}
	public void componentResized(ComponentEvent evt) {
		pixelWidth = getSize().width/resX;
		pixelHeight = getSize().height/resY;
		width = getSize().width;
		height = getSize().height;
    }
    public void mousePressed(MouseEvent evt) {
		if(evt.getButton() == MouseEvent.BUTTON1) {
			RMB = true;
		}
	}
	public void mouseReleased(MouseEvent evt) {
		if(evt.getButton() == MouseEvent.BUTTON1) {
			RMB = false;
		}
		timeOfRelease = System.currentTimeMillis();
	}
	public void mouseMoved(MouseEvent evt) {
		arrayX = (int)(evt.getX()/pixelWidth);
		arrayY = (int)(evt.getY()/pixelHeight);
	}
	public void componentHidden(ComponentEvent evt) {
		
	}
	public void componentMoved(ComponentEvent evt) {
		
	}
	public void componentShown(ComponentEvent evt) {
		
	}
	public void mouseEntered(MouseEvent evt) {
		
	}
	public void mouseExited(MouseEvent evt) {
		
	}
	public void mouseClicked(MouseEvent evt) {
		
	}
	public void mouseDragged(MouseEvent evt) {
		arrayX = (int)(evt.getX()/pixelWidth);
		arrayY = (int)(evt.getY()/pixelHeight);
		if(arrayX <= pixelArray.length) {
			if(arrayY <= pixelArray[arrayX].length) {
				if(RMB) {
					pixelArray[arrayX][arrayY] = 1;
					if(arrayX+1<pixelArray.length) {
						pixelArray[arrayX+1][arrayY] = pixelArray[arrayX][arrayY];
					}
					if(arrayX-1>=0) {
						pixelArray[arrayX-1][arrayY] = pixelArray[arrayX][arrayY];
					}
					if(arrayY+1<pixelArray[arrayX].length) {
						pixelArray[arrayX][arrayY+1] = pixelArray[arrayX][arrayY];
					}
					if(arrayY-1>=0) {
						pixelArray[arrayX][arrayY-1] = pixelArray[arrayX][arrayY];
					}
				} else {
					pixelArray[arrayX][arrayY] = 0;
					if(arrayX+1<pixelArray.length) {
						pixelArray[arrayX+1][arrayY] = pixelArray[arrayX][arrayY];
					}
					if(arrayX-1>=0) {
						pixelArray[arrayX-1][arrayY] = pixelArray[arrayX][arrayY];
					}
					if(arrayY+1<pixelArray[arrayX].length) {
						pixelArray[arrayX][arrayY+1] = pixelArray[arrayX][arrayY];
					}
					if(arrayY-1>=0) {
						pixelArray[arrayX][arrayY-1] = pixelArray[arrayX][arrayY];
					}
				}
				lastX = arrayX;
				lastY = arrayY;
			}
		}
	}
	public void clear() {
		for(curX2 = 0;curX2<pixelArray.length;curX2++) {
			for(curY2 = 0;curY2<pixelArray[curX2].length;curY2++) {
				pixelArray[curX2][curY2] = 0;
			}
		}
		lastX = 0;
		lastY = 0;
	}
}
class neuralNet {
	public static paintPanel drawingSurface;
	public static String dataFolder;
	public static File dataDir;
	public static boolean saveCharacters(learnMethod inputChars[]) {
		RandomAccessFile file;
		int curInput = 0;
		int curIteration = 0;
		char nameChar;
		int offset;
		char lastNameChar = '\0';
		String filePaths[];
		int curFile;
		for(int curChar = 0;curChar<inputChars.length;curChar++) {
			nameChar = Character.toChars((int)(inputChars[curChar].outputs[0]))[0];
			if(nameChar == lastNameChar) {
				curIteration++;
			} else {
				curIteration = 0;
			}
			lastNameChar = nameChar;
			dataDir=new File(dataFolder);
			filePaths = dataDir.list();
			offset = 0;
			for(curFile = 0;curFile<filePaths.length;curFile++) {
				if(filePaths[curFile].contains(String.valueOf(nameChar))) {
					offset++;
				}
			}
			try{
				file = new RandomAccessFile(dataFolder+Character.toChars((int)(inputChars[curChar].outputs[0]))[0]+"-"+(curIteration+offset),"rw");
				for(curInput = 0;curInput<1024;curInput++) {
					file.writeDouble(inputChars[curChar].inputs[curInput]);
					//file.write('\n');
				}
				file.close();
			} catch (Exception e) {
				System.out.println("Error writing "+dataFolder+Character.toChars((int)(inputChars[curChar].outputs[0]))[0]+(curIteration+offset));
				System.out.println(e.getMessage());
			}
		}
		return true;
	}
	public static boolean closingWindow = false;
	public static learnMethod[] getCharacters() {
		char characters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".toCharArray();
		int iterationsPerCharacter = 2;
		learnMethod newMethods[] = new learnMethod[iterationsPerCharacter*characters.length];
		JFrame learningWindow = new JFrame("Learning Window");
		learningWindow.setSize(300,300);
		learningWindow.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
		WindowListener closeOperation = new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				if(!closingWindow) {
					int closeOption = JOptionPane.showOptionDialog(null, "Are you sure you would like to stop the neural network?","Close Confirmation", JOptionPane.YES_NO_OPTION,JOptionPane.QUESTION_MESSAGE, null, null, null);;
					if(closeOption == 0) {
						System.exit(0);
						return;
					}
				}
			}
		};
		learningWindow.addWindowListener(closeOperation);
		JPanel mainPanel = new JPanel();
		mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));
		drawingSurface = new paintPanel(32,32);
		mainPanel.add(drawingSurface);
		JPanel inputPanel = new JPanel();
		inputPanel.setLayout(new BoxLayout(inputPanel, BoxLayout.X_AXIS));
		JButton clearButton = new JButton("Clear");
		clearButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				drawingSurface.clear();
			}
		});
		JButton sendButton = new JButton("Submit");
		sendButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				drawingSurface.getDrawing=true;
				drawingSurface.clear();
			}
		});
		inputPanel.add(clearButton);
		inputPanel.add(sendButton);
		mainPanel.add(inputPanel);
		JLabel characterLabel = new JLabel("Character to draw:" + characters[0] + " (1/" + iterationsPerCharacter + ")"); 
		mainPanel.add(characterLabel);
		learningWindow.add(mainPanel);
		learningWindow.setVisible(true);
		int currentMethod = 0;
		int curX = 0;
		int curY = 0;
		int currentIteration = 1;
		char currentCharacter = characters[0];
		while(currentMethod<newMethods.length) {
			drawingSurface.repaint();
			if(drawingSurface.isReady) {
				if(drawingSurface.getDrawing==true) {
					currentCharacter = characters[(int)((currentMethod)/iterationsPerCharacter)];
					newMethods[currentMethod] = new learnMethod();
					newMethods[currentMethod].inputs = new double[1024];
					newMethods[currentMethod].outputs = new double[1];
					newMethods[currentMethod].outputs[0] = (double)((int)currentCharacter);
					drawingSurface.getDrawing=false;
					for(curX = 0;curX<drawingSurface.pixelArray.length;curX++) {
						for(curY = 0;curY<drawingSurface.pixelArray[curX].length;curY++) {
							newMethods[currentMethod].inputs[(curX*drawingSurface.pixelArray.length)+curY] = drawingSurface.pixelArray[curX][curY];
						}
					}
					currentMethod++;
					if(currentMethod<characters.length) {
						characterLabel.setText("Character to draw:" + characters[((int)((currentMethod)/iterationsPerCharacter))] + " (" + (currentIteration+1) + "/" + (iterationsPerCharacter) + ")");
					} else {
						characterLabel.setText("Character to draw:" + characters[((int)((currentMethod)/iterationsPerCharacter))-1] + " (" + (currentIteration+1) + "/" + (iterationsPerCharacter) + ")");
					}
					currentIteration++;
					if(currentIteration >= iterationsPerCharacter) {
						currentIteration = 0;
					}				
				}
			}
		}
		learningWindow.dispose();
		closingWindow=true;
		learningWindow.dispatchEvent(new WindowEvent(learningWindow, WindowEvent.WINDOW_CLOSING));
		saveCharacters(newMethods);
		return getCharactersFromFolder(dataFolder);
	}
	public static learnMethod[] getCharactersFromFolder(String folderName) {
		dataDir=new File(dataFolder);
		String filePaths[] = dataDir.list();
		RandomAccessFile file;
		int curLine = 0;
		learnMethod newMethods[] = new learnMethod[filePaths.length];
		for(int curMethod = 0;curMethod<newMethods.length;curMethod++) {
			System.out.println("Loading "+dataFolder+filePaths[curMethod]+"...");
			newMethods[curMethod] = new learnMethod();
			newMethods[curMethod].outputs = new double[1];
			newMethods[curMethod].inputs = new double[1024];
			newMethods[curMethod].outputs[0] = filePaths[curMethod].charAt(0);
			try {
				file = new RandomAccessFile(dataFolder+filePaths[curMethod],"r");
				for(curLine = 0;curLine<1024;curLine++) {
					newMethods[curMethod].inputs[curLine] = file.readDouble();
				}
				file.close();
			} catch (Exception e) {
				System.out.println("Error reading "+dataFolder+filePaths[curMethod]+"...");
				System.out.println(e.getMessage());
			}
		}
		return newMethods;
	}
	public static void main(String args[]) {
		double startTime = System.currentTimeMillis();
		if(System.getProperty("os.name").contains("Windows")) {
			dataFolder = "data\\";
		} else {
			dataFolder = "data/";
		}
		Random random = new Random();
		int structure[] = {1024,1280,1280,1280,1280,1280,1}; //Overall Structure
		//CURRENTLY LEARNING CHARACTER RECOGNITION
		System.out.println("Getting learning methods...");
		learnMethod learningMethods[] = getCharacters();
		
		//NOTE: Add option to add initial weights to start from last learning session?
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
		learnThread.threadLimit = 4;
		double desiredTotalError = 0.1;
		double initialDesiredTotalError = 0.9;
		double desiredImprovement = 0.000001; //At least this much
		int improvementLimit = 5000; //Can miss this many times
		double learningConstant = 1.f;
		FFANN idealNetwork = ANNTools.createFFANN(structure);
		double updateRate = 100;
		double curTime = System.currentTimeMillis();
		double lastTime = System.currentTimeMillis();
		System.out.println("Learning CHARACTER RECOGNITION to error at or below " + desiredTotalError + " with " + learnThread.threadLimit + " parallel neural networks...");
		while(learnThread.threadCount<(learnThread.threadLimit+1)) {
			//System.out.println(learnThread.threadLimit);
			new learnThread(learningMethods,initialDesiredTotalError,desiredImprovement,improvementLimit,learningConstant,structure);
		}
		while(learnThread.lowestError>desiredTotalError) {
			curTime = System.currentTimeMillis();
			if((curTime-lastTime)>=updateRate) {;
				lastTime=curTime;
				//Update here if wished.
			}
			if(idealNetwork.getWeights() != learnThread.lastSuccesfulWeights) {
				idealNetwork.setWeights(learnThread.lastSuccesfulWeights);
				//Or update here.
			}
			while(learnThread.threadCount<(learnThread.threadLimit+1)) {
				new learnThread(learningMethods,learnThread.averageNetworkError,desiredImprovement,improvementLimit,learningConstant,structure);
			}
		}
		System.out.println("Learned CHARACTER RECOGNITION with error of " + learnThread.lowestError + " useing " + learnThread.threadLimit + " parallel feed foreward neural networks in " + ((System.currentTimeMillis()-startTime)/1000) + " seconds!");
		System.out.println("Recycled neural networks: " + learnThread.recycledThreads);
		System.out.println("Total generations: " + learnThread.totalGenerations);
		System.out.println("Ideal network structure:");
		System.out.println(" Layers:");
		for(int cl = 0;cl<structure.length;cl++) {
			if(cl == 0) {
				System.out.println("  Input layer: " + structure[cl] + " neurons");
			} else if (cl == (structure.length-1)) {
				System.out.println("  Output layer: " + structure[cl] + " neurons");
			} else {
				System.out.println("  Hidden layer " + cl + ": " + structure[cl] + " neurons");
			}
		}
		double clb[] = idealNetwork.getBiasValues();
		System.out.println(" Bias values:");
		for(int cb = 0;cb<clb.length;cb++) {
			System.out.println("  Bias " + cb + ": " + clb[cb]);
		}
		System.out.println(" Weights:");
		double clw[] = idealNetwork.getWeights();
		for(int cw = 0;cw<clw.length;cw++) {
			System.out.println("  Weight " + cw + ": " + clw[cw]);
		}
		System.out.println("Performance:");
		String inputsString = "";
		String outputsString = "";
		double tTime = 0;
		double sTime = 0;
		double fTime = 0;
		double outputValues[];
		int curInpt = 0;
		int curOut = 0;
		for(int curLearn = 0;curLearn<learningMethods.length;curLearn++) {
			inputsString = "";
			for(curInpt = 0;curInpt<learningMethods[curLearn].inputs.length;curInpt++) {
				inputsString+=learningMethods[curLearn].inputs[curInpt];
				if(curInpt != (learningMethods[curLearn].inputs.length-1)) {
					inputsString+=",";
				}
			}
			System.out.println(" Inputs: " + inputsString);
			sTime = System.currentTimeMillis();
			outputValues = idealNetwork.forwardPass(learningMethods[curLearn].inputs);
			fTime = System.currentTimeMillis();
			tTime = fTime-sTime;
			outputsString = "";
			for(curOut = 0;curOut<outputValues.length;curOut++) {
				outputsString+=outputValues[curOut];
				if(curOut != (outputValues.length-1)) {
					outputsString+=",";
				}
			}
			System.out.println(" Outputs: " + outputsString);
			System.out.println(" Run time: " + (tTime/1000) + " seconds");
		}
	}
}
