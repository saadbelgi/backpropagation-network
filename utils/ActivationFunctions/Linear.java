package utils.ActivationFunctions;

public class Linear implements ActivationFunction {
    private double scale;
    public Linear(double scale) {
        this.scale = scale;
    }
    public Linear() {
        this.scale = 1;
    }

    @Override
    public double transfer(double x) {
        return scale * x;
    }

    @Override
    public double derivative(double x) {
        return scale;
    }
}
