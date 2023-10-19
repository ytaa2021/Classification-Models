package ml.classifiers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

import ml.classifiers.Classifier;
import ml.classifiers.ClassifierFactory;
import ml.data.DataSet;
import ml.data.Example;


/**
 * Represents an implementation of the AVA (All-vs-All) classifier, 
 * which breaks multi-class classification problems into binary classification problems.
 * 
 * @author Tal Mordokh + Yotam Twersky
 */
public class AVAClassifier implements Classifier{
    private ClassifierFactory factory;
    private HashMap<Double[], Classifier> classMap = new HashMap<Double[], Classifier>();
    private double outConfidence;
    private Set<Double> labelMap;


    /**
     * Initializes the AVAClassifier with a given classifier factory
     * 
     * @param factory the factory used to produce classifiers for each classification problem
     */
    public AVAClassifier(ClassifierFactory factory){
        this.factory = factory;
    }

    
    /**
     * creates a new dataset based on the given labels and data
     * 
     * @param label1 the first label
     * @param label2 the second label
     * @param data the original dataset
     * @return a new dataset where instances of label1 are labeled as 1 and instances of label2 are labeled as -1
     */
    public DataSet createNewDataSet(double label1, double label2, DataSet data) {
        DataSet outData = new DataSet(data.getFeatureMap());
        Example addExample;

        //Setting label 1 to 1 and label 2 to -1, if neither, then no add
        for(Example ex : data.getData()) {
            addExample = new Example(ex);
            if(ex.getLabel() == label1){
                addExample.setLabel(1);
            }
            else if (ex.getLabel() == label2) {
                addExample.setLabel(-1);
                outData.addData(addExample);
            }
        }
        return outData;   
    }


    /**
     * trains the classifier on the given dataset
     * 
     * @param data the dataset
     */
    @Override
    public void train(DataSet data) {
        
        labelMap = data.getLabels();
        DataSet currDSet;
        Set<Double> labels = data.getLabels();
        ArrayList<Double> list = new ArrayList<Double>(labels); 

        //creates new datasets, trains on them, and stores the classifiers
        for(int i = 0; i < list.size(); i++ ){
            for (int j = i + 1; j < list.size(); j++) {
                currDSet = createNewDataSet(list.get(i), list.get(j), data);
                Classifier currclassifier = factory.getClassifier();
                currclassifier.train(currDSet);
                Double[] labelArr = {list.get(i), list.get(j)};
                classMap.put(labelArr, currclassifier);
            }
        }
    }


    /**
     * Classifies a given example.
     * 
     * @param example the instance to classify.
     * @return the predicted label for the instance.
     */
    @Override
    public double classify(Example example) {
        ArrayList<Double[]> labels = new ArrayList<Double[]>();
        double[] confidences = new double[classMap.keySet().size()];
        double[] classifications = new double[classMap.keySet().size()];
        HashMap<Double, Double> scores = new HashMap<Double, Double>();

        //initialize all labels to 0
        for (Double key : labelMap){
            scores.put(key, 0.0);
        }
        int i = 0;

        //goes through all classifiers
        for(Double[] labels12 : classMap.keySet()){
            labels.add(labels12);
            confidences[i] = classMap.get(labels12).confidence(example);
            classifications[i] = classMap.get(labels12).classify(example);
            double indexi = labels12[0];
            double indexj = labels12[1];
            double y = confidences[i]*classifications[i];
            
            //stores and aggragates the positive classifications' confidences
            if (y > 0) {
                scores.put(indexi, scores.get(labels12[0]) + confidences[i]);
                scores.put(indexj, scores.get(labels12[1]) - confidences[i]);

            //stores and aggragates the negative classifications' confidences
            } else if (y < 0) {
                scores.put(indexi, scores.get(labels12[0]) - confidences[i]);
                scores.put(indexj, scores.get(labels12[1]) + confidences[i]);
            } 
            i++;
        }

        //finds the correct label
        double maxKey = 0;
        double maxValue = 0.0;
        for(double key : scores.keySet()){
            if (maxValue < scores.get(key)){
                maxKey = key;
                maxValue = scores.get(key);
            }
        }

        return maxKey;
    }


    /**
     * returns the confidence of the classification for a given example
     * currently this method is not fully implemented and will always return 0
     * 
     * @param example the instance to get the confidence for
     * @return the confidence of the classification
     */
    @Override
    public double confidence(Example example) {
        //classify(example);
        //return outConfidence;
        return 0;
    }

}
