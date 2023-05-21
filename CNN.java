package javaML_CNN;

import java.io.IOException;
import java.util.Random;

public class CNN {
	
	final int[] numInp = {28, 28}; //{Y, X}
	double[][] inputs = new double[numInp[0]][numInp[1]];
	final int totalOutputs = 10;
	int totalBatches = 1000000;
	
	final int kernelLayers = 2; 
	final int maxKernels = 8;
	final int[] kernelsPerLayer = {2, maxKernels}; //Account for new channels as well; This is also total no of channels per layer
	final int[][] kernelDimPerLayer = {{5,5}, {3,3}};
	
	double[][][][] kernel = new double[kernelLayers][maxKernels][kernelDimPerLayer[0][0]][kernelDimPerLayer[0][0]]; //Enter Max Dimension
	double[][] kernelBias = new double[kernelLayers][maxKernels];
	
	double[][][][] kernelGRAD = new double[kernelLayers][maxKernels][kernelDimPerLayer[0][0]][kernelDimPerLayer[0][0]];
	double[][] kernelBiasGRAD = new double[kernelLayers][maxKernels];
	
	double[][][][] corrImgs = new double[kernelLayers][maxKernels][numInp[0]][numInp[1]];
	int[][] prevCorrImgsDim = new int[kernelLayers + 1][2];
	
	double[] flatten;
	
	Fully_Connected fcl;
	Operations o;
	
	double step = 0.0;
	
	public CNN() {
		
		fcl = new Fully_Connected(3872, 1, 72, 10);
		o = new Operations();
		
		step = fcl.step;
		
		Random r = new Random();
		
		for(int i = 0; i < kernelGRAD.length; i++) {
			
			for(int j = 0; j < kernelGRAD[0].length; j++) {
				
				for(int k = 0; k < kernelGRAD[0][0].length; k++) {
					
					for(int l = 0; l < kernelGRAD[0][0][0].length; l++) {
						
						kernel[i][j][k][l] = Math.exp(-((k-1)*(k-1) + (l-1)*(l-1)));
						kernel[i][j][k][l] = 0.1;
						kernel[i][j][k][l] = r.nextDouble();
						
					}
					
				}
				
			}
			
		}
		
		initialize();
		
	}
	
	public void initialize() {
		
		double[][] trainingData = null;
		int[] trainingDataLabel = null;

		MnistReader mr = new MnistReader();
		Random r = new Random();

		try {
			trainingData = mr.readData();
			trainingDataLabel = mr.readDataLabel();

		} catch (IOException e) {

			e.printStackTrace();
			System.err.print("Closing");
			System.exit(1);

		}
		
		for (int i = 0; i < totalBatches; i++) {

			int item = r.nextInt(trainingData.length);
			int iter = 0;
			
			for(int j = 0; j < inputs.length; j++) {
				
				for(int k = 0; k < inputs[0].length; k++) {
					
					inputs[j][k] = trainingData[item][iter]; 
					iter++;
						
				}
				
			}
			
			forward();
			
			//Pass To Fully Connected Layer
			double[] desiredOuts = new double[totalOutputs];
			desiredOuts[trainingDataLabel[item]] = 1.0;
			
			fcl.train(flatten, desiredOuts);
			
			if (i % 100 == 0) {

				System.out.println(
						i + " batches completed, batch accuracy: " + fcl.accuracy + " (" + 100 * fcl.accuracy / 100 + "%)");
				fcl.accuracy = 0;
				
				if (i % 1000 == 0) {
					
					for(int i1 = 0; i1 < kernelGRAD.length; i1++) {
						
						for(int j = 0; j < kernelsPerLayer[i1]; j++) {
							
							for(int k = 0; k < kernelGRAD[0][0].length; k++) {
								
								for(int l = 0; l < kernelGRAD[0][0][0].length; l++) {
									
									System.out.println(kernel[i1][j][k][l]);
									
								}
								
							}
							
							System.out.println("BIAS:");
							System.out.println(kernelBias[i1][j]);
							System.out.println("-----------------");
							
						}
						
					}
					
				}
				
			}
			
			kernelGRAD = new double[kernelLayers][maxKernels][kernelDimPerLayer[0][0]][kernelDimPerLayer[0][0]];
			kernelBiasGRAD = new double[kernelLayers][maxKernels];
			
			fcl.getCNNValues(inputs, kernel, kernelBias, kernelsPerLayer, kernelDimPerLayer, corrImgs, prevCorrImgsDim);
			fcl.BackProp();
			
			kernelGRAD = fcl.retCNNkernalGRAD();
			kernelBiasGRAD = fcl.retCNNBiasGRAD();
			
			fcl.descent();
			descent();
			
	//		System.out.println("Completed Batch: " + i);
			
		}
		
	}
	
	public void forward() {
		
		prevCorrImgsDim[0][0] = numInp[0];
		prevCorrImgsDim[0][1] = numInp[1];
		
		//Forward Correlations
		for(int i = 0; i < kernelLayers; i++) {
			
			int sChannel = 0;
			
			for(int j = 0; j < kernelsPerLayer[i]; j++) {
				
				if(i != 0 && j != 0 && j % (kernelsPerLayer[i] / kernelsPerLayer[i - 1]) == 0)
					sChannel++;
				
				if(i == 0) {
					corrImgs[i][j] = correlate(inputs, kernel[i][j], kernelBias[i][j], 
							prevCorrImgsDim[i], kernelDimPerLayer[i], true);
				
				} else {
					
					corrImgs[i][j] = correlate(corrImgs[i - 1][sChannel], kernel[i][j], kernelBias[i][j], 
							prevCorrImgsDim[i], kernelDimPerLayer[i], true);
				}
				
				prevCorrImgsDim[i + 1][0] = prevCorrImgsDim[i][0] - kernelDimPerLayer[i][0] + 1;
				prevCorrImgsDim[i + 1][1] = prevCorrImgsDim[i][1] - kernelDimPerLayer[i][1] + 1;
				
			}
			
		}
		
		//Flattening Last Layer
		flatten = new double[maxKernels * prevCorrImgsDim[prevCorrImgsDim.length - 1][0] * prevCorrImgsDim[prevCorrImgsDim.length - 1][1]];
		
		int flt = 0;
		
		for(int i = 0; i < maxKernels; i++) {
			
			for(int j = 0; j < prevCorrImgsDim[prevCorrImgsDim.length - 1][0]; j++) {
				
				for(int k = 0; k < prevCorrImgsDim[prevCorrImgsDim.length - 1][1]; k++) {
					
					flatten[flt] = corrImgs[kernelLayers - 1][i][j][k];
					flt++;
					
				}
				
			}
			
		}
		
	}
	
	public void descent() {
		
		for(int i = 0; i < kernelGRAD.length; i++) {
			
			for(int j = 0; j < kernelGRAD[0].length; j++) {
				
				kernelBias[i][j] += step * kernelBiasGRAD[i][j];//TODO
				
				for(int k = 0; k < kernelGRAD[0][0].length; k++) {
					
					for(int l = 0; l < kernelGRAD[0][0][0].length; l++) {
						
						kernel[i][j][k][l] += step * kernelGRAD[i][j][k][l];
						
					}
					
				}
				
			}
			
		}
		
	}
	
	//Cross Correlation + Bias Addition + Activation Function
	public double[][] correlate(double[][] inpImg, double[][] kern, double bias, int[] inpImgDim, int[] kernDim, boolean ReLU_OR_SIGMOID) {
		
		int newDimY = inpImgDim[0] - kernDim[0] + 1;
		int newDimX = inpImgDim[1] - kernDim[1] + 1;
		
		double[][] outImg = new double[newDimY][newDimX];
		
		for(int i = 0; i < outImg.length; i++) {
			
			for(int j = 0; j < outImg[0].length; j++) {
				
				for(int k = 0; k < kernDim[0]; k++) {
					
					for(int l = 0; l < kernDim[1]; l++) {
						
						outImg[i][j] += inpImg[i + k][j + l] * kern[k][l];
						
					}
					
				}
				
				outImg[i][j] += bias;
				
				if(ReLU_OR_SIGMOID)
					o.ReLU(outImg[i][j]);
				else
					o.sigmoid(outImg[i][j]);
				
			}
			
		}
		
		return outImg;
		
	}
	
}