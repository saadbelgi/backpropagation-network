package utils.ActivationFunctions;

public final class Ramp implements ActivationFunction {
    public double transfer(double x) {
        if (x < 0)
            return 0;
        if (x > 1)
            return 1;
        return x;
    }

    @Override
    public double derivative(double x) {
        return x > 0 && x < 1 ? 1 : 0;
    }
}