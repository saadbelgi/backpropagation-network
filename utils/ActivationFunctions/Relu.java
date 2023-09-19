package utils.ActivationFunctions;

public final class Relu implements ActivationFunction {
    public double transfer(double x) {
        if (x <= 0)
            return 0;
        return x;
    }

    @Override
    public double derivative(double x) {
        return x <= 0 ? 0 : 1;
    }
}