import java.util.List;

public class GradientDescent {

    public static final String DATA_FILE = "data/ml-assignment-q2.csv";

    public static void main(String[] args) {

        // -------------------------------------------------
        // Data and Graph setup.
        // -------------------------------------------------
        List<List<Double>> data = Data.dataFrom(DATA_FILE);
        Plot plt = new Plot("x & y Classification", "x", "y", data);
        sleep(500);

        // -------------------------------------------------
        // Gradient Descent
        // -------------------------------------------------
        final int epochs = 200;  // Number of iterations we want to run through the algorithm

        // We want to predict h(x) = w1 * x + w0
        double w2 = 0;
        double w1 = 0;
        double w0 = 0;

        // Learning rate
        double alpha = 0.1;

        // Main Gradient Descent Function for Linear Regression
        for(int i = 0; i < epochs; i++) {

            double cost = 0;

            for(int j = 0; j < data.get(0).size(); j++) {

                double x1_j = data.get(0).get(j);
                double x2_j = data.get(1).get(j);
                double y_j = data.get(2).get(j);

                double prediction = 1 / (1 + Math.exp(-(w2 * x2_j + w1 * x1_j + w0)));

                // Cost is based on classification, therefore use an if condition to determine which cost to use
                if (y_j == 1) {
                	cost += Math.log(w2 * x2_j + w1 * x1_j + w0);
                }

                else if (y_j == 0) {
                	cost += Math.log(1 - (w2 * x2_j + w1 * x1_j + w0));
                }

                // Update the parameters for our equation.
                w2 += alpha * (y_j - prediction) * x2_j;
                w1 += alpha * (y_j - prediction) * x1_j;
                w0 += alpha * (y_j - prediction);

            }

            System.out.println("Current Cost: " + cost);


            // ---------------------------------------------
            // Our Hypothesis Function after the epoch
            // (these values are final because of how
            // functional programming works in Java).
            final double w_2 = w2;
            final double w_1 = w1;
            final double w_0 = w0;
            HypothesisFunction h_x = (x) -> (-w_1 * x - w_0)/w_2;
            // ----------------------------------------------
            // Plotting prediction with current values of w
            plt.updatePlot(h_x);
            sleep(50);
            // ----------------------------------------------
        }


        System.out.println("Final Equation: h(x) = (" + w1 + " * x) + " + w0);
    }

    static void sleep(int ticks) {
        try{ Thread.sleep(ticks); } catch(Exception e) { e.printStackTrace(); }
    }
}
