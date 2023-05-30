package javaML_CNN;

import java.io.IOException;
import java.util.Random;

public class Fully_Connected {

	int numHidsNeur;
	
	double[] inputs;
	double[][] hiddenNeurons; // if 0 Layers then set same number of hidden neurons as input
	double[] outputs;

	double[] desiredOuts;

	// WEIGHTS
	double[][] inpTOhid;
	double[][] inpTOhidGRAD;
	
	double[][][] hidTOhid;
	double[][][] hidTOhidGRAD;

	double[][] hidTOout;
	double[][] hidTOoutGRAD;

	// BIAS
	double[][] biasHid;
	double[][] biasHidGRAD;

	double[] biasOut;
	double[] biasOutGRAD;

	// VALUES
	double step = -1.0E-4;
	
	int maxOne = 0;
	double accuracy = 0;
	boolean inTraining = true;

	// Dynamic Step
	boolean dynStepEnable = false;
	double stepLimit = -0.001;
	double prevAccDS = 0.0;
	double changeThresPERC = -0.1;

	Operations operation = new Operations();

	public Fully_Connected(int numInp, int numHidLay, int numHidsNeu, int numOut) {
		
		numHidsNeur = numHidsNeu;
		
		inputs = new double[numInp];
		hiddenNeurons = new double[numHidLay][numHidsNeu];
		outputs = new double[numOut];
		
		desiredOuts = new double[outputs.length];
		
		inpTOhid = new double[numHidsNeu][inputs.length];
		inpTOhidGRAD = new double[numHidsNeu][inputs.length];
		
		if(numHidLay != 0) {
			hidTOhid = new double[numHidLay - 1][hiddenNeurons[0].length][hiddenNeurons[0].length];
			hidTOhidGRAD = new double[numHidLay - 1][hiddenNeurons[0].length][hiddenNeurons[0].length];
		}
		
		hidTOout = new double[outputs.length][numHidsNeu];
		hidTOoutGRAD = new double[outputs.length][numHidsNeu];

		biasHid = new double[numHidLay][numHidsNeu];
		biasHidGRAD = new double[numHidLay][numHidsNeu];
		
		biasOut = new double[outputs.length];
		biasOutGRAD = new double[outputs.length];

		Random r = new Random();

		for (int i = 0; i < inpTOhid.length; i++) {

			for (int j = 0; j < inpTOhid[0].length; j++) {

				inpTOhid[i][j] = r.nextDouble();

			}

		}

		for (int i = 0; hiddenNeurons.length != 0 && i < hidTOhid.length; i++) {

			for (int j = 0; j < hidTOhid[0].length; j++) {

				for (int k = 0; k < hidTOhid[0][0].length; k++) {

					hidTOhid[i][j][k] = r.nextDouble();

				}

			}

		}

		for (int i = 0; i < hidTOout.length; i++) {

			for (int j = 0; j < hidTOout[0].length; j++) {

				hidTOout[i][j] = r.nextDouble();

			}

		}
		
	}

	public void train(double[] inp, double[] desired) {
		
		inputs = inp;
		desiredOuts = desired;
		
		forward();
		
		hidTOoutGRAD = new double[outputs.length][numHidsNeur];
		inpTOhidGRAD = new double[numHidsNeur][inputs.length];
		
		if(hiddenNeurons.length != 0)
			hidTOhidGRAD = new double[hiddenNeurons.length - 1][numHidsNeur][numHidsNeur];
		
		biasHidGRAD = new double[hiddenNeurons.length][numHidsNeur];
		biasOutGRAD = new double[outputs.length];
		
	}

	public void forward() {

		for (int i = 0; hiddenNeurons.length != 0 && i < hiddenNeurons[0].length; i++) {

			hiddenNeurons[0][i] = operation.ReLU(operation.dot(inputs, inpTOhid[i]) + biasHid[0][i]);

		}

		for (int i = 1; hiddenNeurons.length != 0 && i < hiddenNeurons.length; i++) {

			for (int j = 0; j < hiddenNeurons[0].length; j++) {
				
				hiddenNeurons[i][j] = operation
						.ReLU(operation.dot(hiddenNeurons[i - 1], hidTOhid[i - 1][j]) + biasHid[i][j]);

			}

		}

		for (int i = 0; i < outputs.length; i++) {

			if(hiddenNeurons.length != 0)
				outputs[i] = operation
						.sigmoid(operation.dot(hiddenNeurons[hiddenNeurons.length - 1], hidTOout[i]) + biasOut[i]);
			
			else
				outputs[i] = operation.sigmoid(operation.dot(inputs, hidTOout[i]) + biasOut[i]);
			
		}

		// Calculate Accuracy
		maxOne = 0;

		for (int i = 0; i < outputs.length; i++) {

			if (outputs[maxOne] < outputs[i]) {

				maxOne = i;

			}

		}

		if (desiredOuts[maxOne] == 1.0 && inTraining) {

			accuracy += 1.0;

		}

	}

	double[][] inps; 
	
	double[][][][] kernel; 
	double[][] kernelBias;
	int[] kernelsPerLayer; 
	int[][] kernelDimPerLayer;
	double[][][][] corrImgs;
	int[][] prevCorrImgsDim;
	
	double[][][][] kernelGRAD;
	double[][] kernelBiasGRAD;
	
	public void getCNNValues(double[][] inp, double[][][][] k, double[][] kBias, int[] kPL, 
			int[][] kDPL, double[][][][] cI, int[][] pCID) {
		
		inps = inp;
		
		kernel = k;
		
		kernelBias = kBias;
		kernelsPerLayer = kPL; 
		kernelDimPerLayer = kDPL;
		corrImgs = cI;
		prevCorrImgsDim = pCID;
		
		kernelGRAD = new double[k.length][k[0].length][k[0][0].length][k[0][0][0].length];
		kernelBiasGRAD = new double[kBias.length][kBias[0].length];
		
	}
	
	// BackProp
	public void BackProp() {
		
		for (int i = 0; i < hidTOout.length; i++) {

			double basis0 = 2.0 * (outputs[i] - desiredOuts[i]);
			biasOutGRAD[i] = basis0;
			
			for (int j = 0; j < hidTOout[0].length; j++) {
				
				if(hiddenNeurons.length != 0) {
					
					hidTOoutGRAD[i][j] += basis0 * hiddenNeurons[hiddenNeurons.length - 1][j];
					
					double basis1 = basis0 * hidTOout[i][j]
							* operation.ReLUDerivative(hiddenNeurons[hiddenNeurons.length - 1][j]);
					//UnReLU value not required for derivative input; as a 0 ReLU implies 0 derivative
					
					biasHidGRAD[biasHidGRAD.length - 1][j] += basis1;
					
					BackPropPrev(hiddenNeurons.length - 2, j, basis1); //- 2 cuz hidTohid Layer

					
				} else {
					
					hidTOoutGRAD[i][j] += basis0 * inputs[j];
					
					double nBasis = basis0 * hidTOout[i][j] * operation.ReLUDerivative(inputs[j]);
					
					int sChannel = j / (prevCorrImgsDim[kernel.length][0] * prevCorrImgsDim[kernel.length][1]);
					kernelBiasGRAD[kernelBiasGRAD.length - 1][sChannel] += nBasis;
					
					BackPropCNN(nBasis, kernel.length - 1, j);
					
				}
				
			}

		}

	}
	
	// BackProp Chain Rule Handler
	public void BackPropPrev(int layer, int lastNeuron, double basis) {

		if (layer >= 0) {

			for (int i = 0; i < hiddenNeurons[0].length; i++) {

				hidTOhidGRAD[layer][lastNeuron][i] += basis * hiddenNeurons[layer][i];

				double nBasis = basis * hidTOhid[layer][lastNeuron][i]
						* operation.ReLUDerivative(hiddenNeurons[layer][i]);
				biasHidGRAD[layer][i] += nBasis;

				BackPropPrev(layer - 1, i, nBasis);

			}

		} else {

			for (int i = 0; i < inputs.length; i++) {
				
				inpTOhidGRAD[lastNeuron][i] += basis * inputs[i];
				
				double nBasis = basis * inpTOhid[lastNeuron][i] * operation.ReLUDerivative(inputs[i]);
				
				int sChannel = i / (prevCorrImgsDim[kernel.length][0] * prevCorrImgsDim[kernel.length][1]);
				kernelBiasGRAD[kernelBiasGRAD.length - 1][sChannel] += nBasis;
				
				BackPropCNN(nBasis, kernel.length - 1, i);
				
			}
			
		}
		
	}
	
	//CNN BackProp AIO
	public void BackPropCNN(double basis, int layer, int pix) {
		
		int sChannel = pix / (prevCorrImgsDim[layer + 1][0] * prevCorrImgsDim[layer + 1][1]);
		
		int pixM = pix % (prevCorrImgsDim[layer + 1][1]);
		int pixD = (pix - (sChannel * prevCorrImgsDim[layer + 1][0] * prevCorrImgsDim[layer + 1][1])) / (prevCorrImgsDim[layer + 1][1]);
		
		if(layer > 0) {
			
			int pChannel = sChannel / (kernelsPerLayer[layer] / kernelsPerLayer[layer - 1]);
			
			for(int i = 0; i < kernelDimPerLayer[layer][0]; i++) {
				
				for(int j = 0; j < kernelDimPerLayer[layer][1]; j++) {
					
					kernelGRAD[layer][sChannel][i][j] += basis * corrImgs[layer - 1][pChannel][i + pixD][j + pixM];
					
					double nBasis = basis * kernel[layer][sChannel][i][j] 
							* operation.ReLUDerivative(corrImgs[layer - 1][pChannel][i + pixD][j + pixM]);
					
					kernelBiasGRAD[layer - 1][pChannel] += nBasis;
					
					BackPropCNN(nBasis, layer - 1, (pChannel * prevCorrImgsDim[layer][0] * prevCorrImgsDim[layer][1]) + 
							((i + pixD) * prevCorrImgsDim[layer][0]) + (j + pixM));
					
				}
				
			}
			
		} else {
			
			for(int i = 0; i < kernelDimPerLayer[layer][0]; i++) {
				
				for(int j = 0; j < kernelDimPerLayer[layer][1]; j++) {
					
					kernelGRAD[layer][sChannel][i][j] += basis * inps[i + pixD][j + pixM];
					
				}
				
			}
			
		}
		
	}

	public void descent() {

		for (int i = 0; i < hidTOout.length; i++) {

			biasOut[i] += step * biasOutGRAD[i];

			for (int j = 0; j < hidTOout[0].length; j++) {

				hidTOout[i][j] += step * hidTOoutGRAD[i][j];

			}

		}

		for (int i = 0; hiddenNeurons.length != 0 && i < hidTOhid.length; i++) {

			for (int j = 0; j < hidTOhid[0].length; j++) {

				biasHid[i + 1][j] += step * biasHidGRAD[i + 1][j];

				for (int k = 0; k < hidTOhid[0][0].length; k++) {

					hidTOhid[i][j][k] += step * hidTOhidGRAD[i][j][k];

				}

			}

		}

		for (int i = 0; hiddenNeurons.length != 0 && i < inpTOhid.length; i++) {

			biasHid[0][i] += step * biasHidGRAD[0][i];

			for (int j = 0; j < inpTOhid[0].length; j++) {

				inpTOhid[i][j] += step * inpTOhidGRAD[i][j];

			}

		}

	}
	
	public double[][][][] retCNNkernalGRAD() {
		
		return kernelGRAD;
		
	}
	
	public double[][] retCNNBiasGRAD() {
		
		return kernelBiasGRAD;
		
	}

	public void dynamicStep() {

		double diff = 0.0;

		if (prevAccDS == 0.0) {

			prevAccDS = 100.0 * accuracy / 10000.0;

		} else {

			diff = (100.0 * accuracy / 10000.0) - prevAccDS;

			if (diff < changeThresPERC) {

				if (step < stepLimit) {

					step *= 0.1;
					changeThresPERC *= 0.1;
					System.out.println("New Step Size: " + step);

				}

			}

			prevAccDS = 0.0;

		}

	}

	public void testSeries() {

		accuracy = 1000;

		double[][] trainingData = null;
		int[] trainingDataLabel = null;

		MnistReader mr = new MnistReader();
		Random r = new Random();
		
		try {
			trainingData = mr.readDataTest();
			trainingDataLabel = mr.readDataLabelTest();

		} catch (IOException e) {

			e.printStackTrace();
			System.err.print("Closing");
			System.exit(1);

		}

		for (int i = 0; i < 1000; i++) {

			int item = r.nextInt(trainingData.length);
			inputs = trainingData[item];
			forward();
			if (maxOne != trainingDataLabel[item]) {
				System.out.print("(MISTAKE)");
				accuracy -= 1;
			}

			System.out.println("Predicted: " + maxOne + " Real: " + trainingDataLabel[item] + " Item: " + item);

		}

		System.out.println(accuracy);

	}

}
