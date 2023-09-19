import utils.Layers.HiddenLayer;
import utils.Layers.OutputLayer;

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

public class BackPropagationNetwork {

    private final int nInput; // no of nodes in input layer
    private final HiddenLayer[] hiddenLayers;
    private final OutputLayer outputLayer;
    private final double learningRate;
    private final String[] availableActivationFunctions = { "Binary sigmoidal", "Bipolar sigmoidal", "tanh", "ramp",
            "relu", "Linear" };

    public BackPropagationNetwork(Scanner sc) {
        // initialization of network architecture
        System.out.print("Enter number of nodes in input layer: ");
        this.nInput = sc.nextInt();
        System.out.print("Enter learning rate [between 0 and 1] : ");
        this.learningRate = sc.nextDouble();
        System.out.print("Enter number of hidden layers: ");
        this.hiddenLayers = new HiddenLayer[Math.abs(sc.nextInt())];
        System.out
                .println("\nFollowing is the list of activation functions available along with their choice codes:\n");
        System.out.println("Choice code | Function name");
        System.out.println("--------------------------------");
        for (int i = 0; i < availableActivationFunctions.length; i++)
            System.out.printf("%-11d | %17s\n", i + 1, availableActivationFunctions[i]);
        int prevLayerSize = this.nInput;
        for (int i = 0; i < this.hiddenLayers.length; i++) {
            System.out.printf("For hidden layer %d:\n", i + 1);
            this.hiddenLayers[i] = new HiddenLayer(prevLayerSize, this.learningRate, sc);
            prevLayerSize = this.hiddenLayers[i].getSize();
        }
        System.out.println("For output layer:");
        this.outputLayer = new OutputLayer(prevLayerSize, this.learningRate, sc);
    }

    public void train(double[][] input, double[][] output, int maxEpochs) throws IOException {
        if (input.length != output.length || input.length == 0 || maxEpochs <= 0 || input[0].length != nInput
                || output[0].length != this.outputLayer.getSize()) {
            return;
        }
        double[] rmseHistory = new double[maxEpochs];
        double[] rmse = new double[output[0].length];
        double[] temp, error;
        double[][] predicted = new double[output.length][output[0].length];
        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            System.out.printf("Epoch %d:\n", epoch);
            for (int i = 0; i < input.length; i++) {
                temp = input[i];
                for (HiddenLayer hl : hiddenLayers) {
                    temp = hl.getOutput(temp);
                }
                predicted[i] = outputLayer.getOutput(temp);
                error = outputLayer.changeWeightsAndPropagateError(output[i]);
                for (int j = hiddenLayers.length - 1; j >= 0; j--) {
                    error = hiddenLayers[j].changeWeightsAndPropagateError(error);
                }
            }
            double overallRMSE = 0;
            for (int i = 0; i < output[0].length; i++) {
                for (int j = 0; j < output.length; j++) {
                    rmse[i] += Math.pow(output[j][i] - predicted[j][i], 2);
                }
                rmse[i] /= output.length;
                overallRMSE += rmse[i];
            }
            overallRMSE /= output[0].length;
            System.out.printf("RMSE: %f\n", overallRMSE);
            rmseHistory[epoch - 1] = overallRMSE;
        }
        // write RMSE history to a file
        BufferedWriter bw = new BufferedWriter(new FileWriter("rmse-history.txt"));
        for (double i : rmseHistory) {
            bw.write(i + "\n");
        }
        bw.close();
    }

    public void test(double[][] input, double[][] output, double[][] minAndScale) {
        if (input.length != output.length || input.length == 0 || input[0].length != nInput
                || output[0].length != this.outputLayer.getSize()) {
            return;
        }
        double[] rmse = new double[output[0].length];
        double[] temp;
        double[][] predicted = new double[output.length][output[0].length];
        for (int i = 0; i < input.length; i++) {
            temp = input[i];
            for (HiddenLayer hl : hiddenLayers) {
                temp = hl.getOutput(temp);
            }
            predicted[i] = outputLayer.getOutput(temp);
        }
        double overallRMSE = 0;
        for (int i = 0; i < output[0].length; i++) {
            for (int j = 0; j < output.length; j++) {
                rmse[i] += Math.pow(output[j][i] - predicted[j][i], 2);
            }
            rmse[i] /= output.length;
            overallRMSE += rmse[i];
        }
        overallRMSE /= output[0].length;
        for (int i = 0; i < output[0].length; i++) {
            System.out.printf("| Predicted-%d  |   Actual-%d   | ", i + 1, i + 1);
        }
        System.out.println();
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[0].length; j++) {
                System.out.printf("| %12f | %12f | ", predicted[i][j] * minAndScale[j][1] + minAndScale[j][0],
                        output[i][j] * minAndScale[j][1] + minAndScale[j][0]);
            }
            System.out.println();
        }
        System.out.printf("\nRMSE: %f\n", overallRMSE);
    }

    public static void main(String[] args) throws IOException {
        // System.setIn(new FileInputStream("input.txt"));
        Scanner sc = new Scanner(System.in);
        BackPropagationNetwork bpn = new BackPropagationNetwork(sc);
        double[][][] data = BackPropagationNetwork.readCsv("C:\\Users\\arifa\\Desktop\\archive\\Folds5x2_pp.csv", 1, 1);
        double[][] input = data[0], output = data[1];
        double[][] normalizedInput = normalizeColumns(input)[0];
        double[][][] temp = normalizeColumns(output);
        double[][] normalizedOutput = temp[0];
        double[][] minAndScale = temp[1];

        // partitioning data
        int trainSize = (int) Math.floor(input.length * 0.8);
        int testSize = input.length - trainSize;
        double[][] trainInputData = new double[trainSize][input[0].length];
        System.arraycopy(normalizedInput, 0, trainInputData, 0, trainSize);
        double[][] trainOutputData = new double[trainSize][output[0].length];
        System.arraycopy(normalizedOutput, 0, trainOutputData, 0, trainSize);
        double[][] testInputData = new double[testSize][input[0].length];
        System.arraycopy(normalizedInput, trainSize, testInputData, 0, testSize);
        double[][] testOutputData = new double[testSize][output[0].length];
        System.arraycopy(normalizedOutput, trainSize, testOutputData, 0, testSize);
        System.out.println("Maximum number of epochs: ");
        int maxEpochs = sc.nextInt();
        System.out.println("Training......");
        bpn.train(trainInputData, trainOutputData, maxEpochs);
        System.out.println("Testing......");
        bpn.test(testInputData, testOutputData, minAndScale);
    }

    public static double[][][] readCsv(String path, int outputCols, int start) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(path));
        ArrayList<ArrayList<double[]>> data = new ArrayList<>();
        data.add(new ArrayList<>()); // input
        data.add(new ArrayList<>()); // output
        String line = "";
        for (int i = 0; i < start; i++)
            br.readLine(); // skip lines
        while ((line = br.readLine()) != null) {
            String[] strs = line.split(",");
            double[] input = new double[strs.length - outputCols];
            for (int i = 0; i < strs.length - outputCols; i++) {
                input[i] = Double.parseDouble(strs[i]);
            }
            double[] output = new double[outputCols];
            for (int i = strs.length - outputCols; i < strs.length; i++) {
                output[i - strs.length + outputCols] = Double.parseDouble(strs[i]);
            }
            data.get(0).add(input);
            data.get(1).add(output);
        }
        br.close();
        return new double[][][] { data.get(0).toArray(double[][]::new), data.get(1).toArray(double[][]::new) };
    }

    public static double[][][] normalizeColumns(double[][] x) {
        double[][] normalized = new double[x.length][x[0].length];
        double[][] minAndScale = new double[x[0].length][2];
        for (int i = 0; i < x[0].length; i++) {
            double min = x[0][i], max = x[0][i];
            for (int j = 1; j < x.length; j++) {
                if (x[j][i] < min)
                    min = x[j][i];
                else if (x[j][i] > max)
                    max = x[j][i];
            }
            double scale = max - min;
            for (int j = 0; j < x.length; j++) {
                normalized[j][i] = (x[j][i] - min) / scale;
            }
            minAndScale[i][0] = min;
            minAndScale[i][1] = scale;
        }
        return new double[][][] { normalized, minAndScale };
    }
}
