package utils.ActivationFunctions;

public final class Tanh implements ActivationFunction {
    public double transfer(double x) {
        return Math.tanh(x);
    }

    @Override
    public double derivative(double x) {
        double fx = Math.tanh(x);
        return (1 + fx) * (1 - fx);
    }
}