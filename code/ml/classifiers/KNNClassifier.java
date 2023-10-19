package ml.classifiers;
import ml.utils.ExampleDistance;
import ml.utils.ExampleDistanceComparator;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;

import ml.data.DataSet;
import ml.data.Example;


/**
 * @author Tal Mordokh anf Yotam Twersky
 * KNNClassifier classifies examples based on the majority label among the k closest examples from the training set
 */
public class KNNClassifier implements Classifier{
    public DataSet data;
     // number of neighbors to consider (default is 3)
    public int k = 3;


    /**
     * trains the KNN classifier
     * 
     * @param data training dataset
     */
    @Override
    public void train(DataSet data) {
        this.data = data;
    }


    /**
     * set the number of neighbors to consider during classification
     * 
     * @param k number of neighbors
     */
    public void setK(int k){
        this.k = k;
    }


    /**
     * classify a given example based on the k-nearest neighbors in the training set
     * 
     * @param example the example to classify
     * @return the predicted class label
     */
    @Override
    public double classify(Example example) {
        double sum;
        ArrayList<ExampleDistance> exampleArr = new ArrayList<ExampleDistance>();

        //calculate distances
        for(Example currEx : data.getData()){
            sum = 0;
            for(int currFeat : example.getFeatureSet()){
                sum = sum + Math.pow(currEx.getFeature(currFeat) - example.getFeature(currFeat), 2);
            }

            //sorting
            ExampleDistance exDist = new ExampleDistance(currEx, Math.sqrt(sum));
            if(exampleArr.size() < k){
                exampleArr.add(exDist);
                exampleArr.sort(new ExampleDistanceComparator());
            }
            else if(exampleArr.get(k - 1).getDistance() > exDist.getDistance()){
                exampleArr.set(k - 1, exDist);
                exampleArr.sort(new ExampleDistanceComparator());
            }

        }

        //counting labels
        Example currEx = new Example();
        HashMap<Double, Double> labels = new HashMap<Double, Double>();
        for(ExampleDistance currExDist : exampleArr){
            currEx = currExDist.getExample();
            if (!labels.containsKey(currEx.getLabel())){
                labels.put(currEx.getLabel(), 1.0);
            }
            else{
                labels.put(currEx.getLabel(), labels.get(currEx.getLabel()) + 1);
            }
        }

        //determining majority label
        double maxKey = 0.0;
        double maxValue = 0.0;
        for(double labelKey : labels.keySet()){
            if(labels.get(labelKey) > maxValue){
                maxValue = labels.get(labelKey);
                maxKey = labelKey;

            }
        }

        return maxKey;
    }

    


    
    
    
}


