package ml.classifiers;

import java.util.ArrayList;
import java.util.HashMap;

import ml.classifiers.Classifier;
import ml.classifiers.ClassifierFactory;
import ml.data.DataSet;
import ml.data.Example;

/**
 * One-Versus-All (OVA) classifier which trains a separate
 * classifier for each class against all other classes
 * 
 * @author Tal Mordokh + Yotam Twersky
 */
public class OVAClassifier implements Classifier{
    private ClassifierFactory factory;
    private HashMap<Double, Classifier> classMap = new HashMap<Double, Classifier>();
    private double outConfidence;


    /**
     * constructor for OVAClassifier
     * @param factory the classifier factory used to create classifiers.
     */
    public OVAClassifier(ClassifierFactory factory){
        this.factory = factory;
    }


    /**
     * creates a new dataset with modified labels based on the given label
     * if the example label matches the given label, its label is set to 1
     * otherwise, it's set to -1
     *
     * @param label the label against which other labels are compared
     * @param data the original dataset
     * @return a new dataset with modified labels
     */
    public DataSet createNewDataSet(double label, DataSet data) {
        DataSet outData = new DataSet(data.getFeatureMap());
        Example addExample;

        //iterating over examples and setting them to 1 if matches our provided label
        for(Example ex : data.getData()) {
            addExample = new Example(ex);
            if(ex.getLabel() != label){
                addExample.setLabel(-1);
            }
            else{
                addExample.setLabel(1);
            }
            outData.addData(addExample);
        }
        return outData;   
    }

    
    /**
     * trains a classifier for each label in the dataset
     * @param data the dataset
     */
    @Override
    public void train(DataSet data) {
        DataSet currDSet;

        //iterating through all labels and creating, storing, and training a corresponding classifier
        for(double label : data.getLabels()){
            currDSet = createNewDataSet(label, data);
            Classifier currclassifier = factory.getClassifier();
            currclassifier.train(currDSet);
            classMap.put(label, currclassifier);
        }

    }


    /**
     * classifies an example using OVA approach. If there's ambiguity, 
     * the classifier will choose the most confident positive class 
     * or the least confident negative class
     *
     * @param example the instance to classify
     * @return predicted label for the given instance.
     */
    @Override
    public double classify(Example example) {
        double[] labels = new double[classMap.keySet().size()];
        double[] confidences = new double[classMap.keySet().size()];
        double[] classifications = new double[classMap.keySet().size()];
        int i = 0;

        // Populate the labels, confidences, and classifications arrays
        for(double label : classMap.keySet()){
            labels[i] = label;
            confidences[i] = classMap.get(label).confidence(example);
            classifications[i] = classMap.get(label).classify(example);
            i++;
        }
        
        double maxPos = 0.0;
        double minNeg = 1.0;
        ArrayList<Double> minNegList = new ArrayList<Double>();
        ArrayList<Double> maxPosList = new ArrayList<Double>();

        // Iterate over classifications to populate above lists
        for (i = 0; i<labels.length; i++) {
             // Handle positive classifications
            if (classifications[i] > 0) {
                if (confidences[i] > maxPos) {
                    maxPosList.clear();
                    maxPos = confidences[i];
                    maxPosList.add(labels[i]);
                }
                else if (confidences[i] == maxPos) {
                    maxPosList.add(labels[i]);
                } 
            }
            // Handle negative classifications
            if (classifications[i] < 0){
                if (confidences[i] < minNeg){
                    minNegList.clear();
                    minNeg = confidences[i];
                    minNegList.add(labels[i]);
                }
                else if (confidences[i] == minNeg) {
                    minNegList.add(labels[i]);
                } 
            }
        }

         // Return the most confident positive classification, or the least confident negative one if no positives found
        if (maxPosList.size() >= 1) {
            outConfidence = maxPos;
            return maxPosList.get(0);
        } 
        
        outConfidence = minNeg;
        return minNegList.get(0);
    }

    public void printClassifierTree(Double label){
        Classifier zing = classMap.get(label);
        System.out.println(zing.toString());
    }


    /**
     * retrieves the confidence of the classification.
     * currently, this method always returns 0
     * 
     * @param example the instance to determine the confidence for
     * @return confidence value
     */
    @Override
    public double confidence(Example example) {
        return 0;
    }

}
