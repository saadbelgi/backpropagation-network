package utils.ActivationFunctions;

public final class BinarySigmoidal extends Sigmoidal {
    public BinarySigmoidal() {
    }

    public BinarySigmoidal(double steepness) {
        this.steepness = steepness;
    }

    public double transfer(double x) {
        return 1 / (1 + Math.exp(-1 * this.steepness * x));
    }

    @Override
    public double derivative(double x) {
        double fx = transfer(x);
        return steepness * fx * (1 - fx);
    }
}