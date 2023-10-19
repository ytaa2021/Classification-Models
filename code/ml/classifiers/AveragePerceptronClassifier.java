package ml.classifiers;
/**
 * @author Tal Mordokh + Yotam Twersky
 * Default constructor that sets the classifier to use averaged learning.
 */
public class AveragePerceptronClassifier extends PerceptronClassifier{
    public AveragePerceptronClassifier(){
        isAvg = true;
    }
}
