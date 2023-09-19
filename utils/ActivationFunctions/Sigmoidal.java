package utils.ActivationFunctions;

public abstract class Sigmoidal implements ActivationFunction {
    protected double steepness = 1;

    public double getSteepness() {
        return this.steepness;
    }

    public void setSteepness(double steepness) {
        this.steepness = steepness;
    }

}
