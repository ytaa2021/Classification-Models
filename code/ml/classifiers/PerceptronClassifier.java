package ml.classifiers;

import ml.Example;

import java.util.ArrayList;
import java.util.Collections;

import ml.DataSet;

/**
 * Interface for a classifier.
 * 
 * @author Tal Mordokh + Yotam Twersky
 *
 */
public class PerceptronClassifier implements Classifier {
    public int iterations;
    double[] weights;
    double[] uWeights;
    double b = 0;
    double b2 = 0;
    boolean isAvg = false;
    DataSet data;


    /**
     * Default constructor setting the number of iterations to 10.
     */
    public PerceptronClassifier() {
        iterations = 10;
    }


	/**
     * Predicts the label for a given example.
     * 
     * @param example the input example
     * @return -1, 0, or 1 based on the model's prediction
     */
    public double predict(Example example) {
        double sum;
        double label;
        double value;
        sum = b;
        label = example.getLabel();
        for (int k = 0; k < data.getAllFeatureIndices().size(); k++) {
            value = example.getFeature(k);
            sum = sum + value*weights[k];
        }
        if (sum < 0){
            return -1;
        }
        else if (sum == 0) {
            return 0;
        } else {
            return 1;
        }

    }


    /**
     * Train the perceptron classifier using the provided dataset.
     * 
     * @param data the dataset for training
     */
	public void train(DataSet data) {
        this.data = data;
        int numFeatures = data.getFeatureMap().size();
        weights = new double[numFeatures];
        uWeights = new double[numFeatures];
       
        //check each example
        //if not correct update all the weights:
        //Weight*feature=value
        
        Example example;
        int k;
        double result;
        double updated = 0;
        double total = 0;
        ArrayList<Example> randData = data.getData();

        // Iterate multiple times over the data to refine the model
        for (int i = 0; i<iterations; i++) {
            Collections.shuffle(randData);
            for (int j = 0; j < randData.size(); j++) {
                example = randData.get(j);
                result = predict(example);

                // Update weights if prediction is incorrect
                if (result*example.getLabel() <= 0) {
                    //case for avg perceptron model
                    if (isAvg){
                        for (k = 0; k < randData.get(0).getFeatureSet().size(); k++){
                            uWeights[k] = uWeights[k] + updated*weights[k];
                        }
                        b2 = b2 + updated*b;
                        updated = 0;
                    }
                    for (k = 0; k < randData.get(j).getFeatureSet().size(); k++) {
                        weights[k] = weights[k] + example.getFeature(k)*example.getLabel();
                    }
                    b = b+example.getLabel();
                }
                //case for avg perceptron model
                if (isAvg){
                    updated++;
                    total++;
                }
            }
        }

        //case for avg perceptron model
        if (isAvg){
            for (k = 0; k < randData.get(0).getFeatureSet().size(); k++){
                uWeights[k] = uWeights[k] + updated*weights[k];
            }
            b2 = b2 + updated*b;
            for (double u_i : uWeights){
                u_i = u_i / total;
            }
            b2 = b2/total;
        }
    }
	

	/**
     * Set the number of iterations for the perceptron training.
     * 
     * @param n the desired number of iterations
     */
    public void setIterations(int n) {
        iterations = n;
    }


    /**
     * Returns a string representation of the perceptron classifier, mainly showing the weights.
     * 
     * @return a string representation of the perceptron classifier
     */
    public String toString() {
        String outString = "";
        for (int i = 0; i < weights.length; i++) {
            outString = outString + i + ":" + weights[i] + " ";
        }
        outString = outString + b;
        return outString;
    }


    /**
     * Classifies a given example using either the averaged model or the non-averaged model.
     * 
     * @param example the input example to classify
     * @return the predicted class label (-1, 0, or 1)
     */
	public double classify(Example example){
        if (isAvg){    
            double value;
            double sum = b2;
            for (int k = 0; k < data.getAllFeatureIndices().size(); k++) {
                value = example.getFeature(k);
                sum = sum + value*uWeights[k];
            }
            if (sum < 0){
                return -1;
            }
            else if (sum == 0) {
                return 0;
            } else {
                return 1;
            }
        }
        return predict(example);
    }
}
