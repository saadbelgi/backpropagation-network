package utils.Layers;

import utils.ActivationFunctions.*;

import java.util.Random;
import java.util.Scanner;

abstract public class BPNLayer {
    protected final int nNodes;
    protected ActivationFunction activationFunction;
    protected final double[][] weightMatrix;
    protected final double[] bias;
    protected final double[] summedInput;
    protected double[] input, output;
    protected final double learningRate;

    public int getSize() {
        return nNodes;
    }

    public BPNLayer(int previousLayerSize, double learningRate, Scanner sc) {
        this.learningRate = learningRate;
        System.out.print("Enter number of nodes: ");
        this.nNodes = sc.nextInt();
        System.out.print("Enter choice code of the desired activation function: ");
        int choice = sc.nextInt();
        switch (choice) {
            case 1 -> {
                System.out.print("Enter steepness value [default 1]: ");
                this.activationFunction = new BinarySigmoidal(sc.nextDouble());
            }
            case 2 -> {
                System.out.print("Enter steepness value [default 1]: ");
                this.activationFunction = new BipolarSigmoidal(sc.nextDouble());
            }
            case 3 -> this.activationFunction = new Tanh();
            case 4 -> this.activationFunction = new Ramp();
            case 5 -> this.activationFunction = new Relu();
            case 6 -> {
                System.out.print("Enter scale [default 1] : ");
                this.activationFunction = new Linear(sc.nextDouble());
            }
        }
        this.weightMatrix = new double[previousLayerSize][this.nNodes];
        this.bias = new double[this.nNodes];
        this.summedInput = new double[this.nNodes];
        // using Xavier weight initialisation
        double lower = -1 / Math.sqrt(previousLayerSize);
        double upper = -1 * lower;
        Random r = new Random(0); // seed = 0 to keep weights constant between different runs
        for (int i = 0; i < previousLayerSize; i++) {
            for (int j = 0; j < nNodes; j++)
                this.weightMatrix[i][j] = r.nextDouble(lower, upper);
        }
        for (int i = 0; i < this.nNodes; i++)
            bias[i] = r.nextDouble(lower, upper);
    }

    public double[] getOutput(double[] input) {
        this.input = input;
        if (input.length != weightMatrix.length)
            return null;
        double[] output = new double[this.nNodes];
        for (int i = 0; i < this.nNodes; i++) {
            this.summedInput[i] = bias[i];
            for (int j = 0; j < weightMatrix.length; j++) {
                this.summedInput[i] += input[j] * weightMatrix[j][i];
            }
            output[i] = this.activationFunction.transfer(summedInput[i]);
        }
        this.output = output;
        return output;
    }

    abstract double[] changeWeightsAndPropagateError(double[] in);
}
