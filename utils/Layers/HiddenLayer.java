package utils.Layers;

import java.util.Scanner;

public class HiddenLayer extends BPNLayer {
    private double[] deltaInput;
    public HiddenLayer(int previousLayerSize, double learningRate, Scanner sc) {
        super(previousLayerSize, learningRate, sc);
    }

    public double[] changeWeightsAndPropagateError(double[] deltaInput) {
        if(deltaInput.length != this.nNodes)
            return null;
        double[] delta = new double[this.nNodes];
        double[] deltaOutput = new double[this.weightMatrix.length];
        // calculating real delta from input delta
        for(int i = 0; i < this.nNodes; i++) {
            delta[i] = deltaInput[i] * this.activationFunction.derivative(this.summedInput[i]);
        }
        // calculating weighted sum of deltas to be sent to the previous layer
        for(int i = 0; i < this.weightMatrix.length; i++) {
            deltaOutput[i] = 0;
            for(int j = 0; j < this.nNodes; j++) {
                deltaOutput[i] += this.weightMatrix[i][j] * delta[j];
            }
        }
        // updating weights and bias
        for(int i = 0; i < this.weightMatrix.length; i++) {
            for(int j = 0; j < this.nNodes; j++) {
                weightMatrix[i][j] += this.learningRate * delta[j] * input[i];
            }
        }
        for(int i = 0; i < this.nNodes; i++) {
            bias[i] = this.learningRate * delta[i];
        }
//        System.out.println();
//        for(double[] i: weightMatrix) {
//            for(double j: i) {
//                System.out.print(j + " ");
//            }
//            System.out.println();
//        }
        return deltaOutput;
    }
}
