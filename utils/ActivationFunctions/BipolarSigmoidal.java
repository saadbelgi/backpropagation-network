package utils.ActivationFunctions;

public final class BipolarSigmoidal extends Sigmoidal {
    public BipolarSigmoidal() {
    }

    public BipolarSigmoidal(double steepness) {
        this.steepness = steepness;
    }

    public double transfer(double x) {
        double temp = Math.exp(-1 * this.steepness * x);
        return (1 - temp) / (1 + temp);
    }

    @Override
    public double derivative(double x) {
        double fx = transfer(x);
        return (1 + fx) * (1 - fx);
    }
}
