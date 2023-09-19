package utils.Layers;

import java.util.Scanner;

public class OutputLayer extends BPNLayer {
    public OutputLayer(int previousLayerSize, double learningRate, Scanner sc) {
        super(previousLayerSize, learningRate, sc);
    }

    public double[] changeWeightsAndPropagateError(double[] target) {
        if (target.length != this.nNodes) {
            return null;
        }
        double[] delta = new double[this.nNodes];
        double[] deltaOutput = new double[this.weightMatrix.length];
        for (int i = 0; i < this.nNodes; i++) {
            delta[i] = (target[i] - output[i]) * this.activationFunction.derivative(summedInput[i]);
        }
        // calculating weighted sum of deltas to be sent to the previous layer
        for (int i = 0; i < this.weightMatrix.length; i++) {
            deltaOutput[i] = 0;
            for (int j = 0; j < this.nNodes; j++) {
                deltaOutput[i] += this.weightMatrix[i][j] * delta[j];
            }
        }
        // updating weights and bias
        for (int i = 0; i < this.weightMatrix.length; i++) {
            for (int j = 0; j < this.nNodes; j++) {
                weightMatrix[i][j] += this.learningRate * delta[j] * input[i];
            }
        }
        for (int i = 0; i < this.nNodes; i++) {
            bias[i] = this.learningRate * delta[i];
        }
        return deltaOutput;
    }
}
