package ml;
import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.Example;
import ml.classifiers.KNNClassifier;
import ml.classifiers.OVAClassifier;
import ml.data.DataSetSplit;
import ml.data.DataSet;
import ml.classifiers.ClassifierTimer;
import ml.classifiers.AVAClassifier;
import ml.classifiers.AveragePerceptronClassifier;
import ml.classifiers.Classifier;
import ml.classifiers.ClassifierFactory;
import ml.classifiers.DecisionTreeClassifier;

/**
 * @author Tal Mordokh and Yotam Twersky
 * 
 * the Experimenter class is designed to run experiments using different classifiers and preprocessors on a dataset.
 * it uses a 10-fold cross-validation setup to assess the classifier's performance
 */
public class Experimenter {
    public static void main(String[] args) {

        //sets classifiers, normalizers, and other variables
        DataSet data = new DataSet("data/wines.train", 1);
        
        //AveragePerceptronClassifier apc = new AveragePerceptronClassifier();
        
        CrossValidationSet crossVal = new CrossValidationSet(data, 10, true);
        double correct = 0.0;
        double incorrect = 0.0;
        double sumov1;
        double sumov2;
        double sumov3;
        double sumav1;
        double sumav2;
        double sumav3;
        double sum22;

        //section7
        double sumov7 = 0.0;
        double correct7 = 0.0;
        double incorrect7 = 0.0;
        ClassifierFactory DT7 = new ClassifierFactory(ClassifierFactory.DECISION_TREE, 3);
        OVAClassifier OVAClassifier7 = new OVAClassifier(DT7);
        OVAClassifier7.train(data);
        /*
        for (int i = 0; i < data.getData().size(); i++)
                    if (OVAClassifier7.classify(data.getData().get(i)) == data.getData().get(i).getLabel()) {
                        correct7++;
                    } else {
                        incorrect7++;
                    }
                sumov7 += correct7/(correct7 + incorrect7);
        */
        Double Zinglabel = 10.0;
        OVAClassifier7.printClassifierTree(Zinglabel);      
        //System.out.println(sumov7); //0.6776349614395887
        
        
        
        //section5

        ClassifierFactory DT35 = new ClassifierFactory(ClassifierFactory.DECISION_TREE, 3);
        OVAClassifier OClassifier3 = new OVAClassifier(DT35);
        AVAClassifier AClassifier3 = new AVAClassifier(DT35);
        ClassifierTimer classtime = new ClassifierTimer();
        System.out.println("OVA D3 time for 5 runs");
        classtime.timeClassifier(OClassifier3, data, 5);
        System.out.println("AVA D3 time for 5 runs");
        classtime.timeClassifier(AClassifier3, data, 5);


        //loop through each fold in the cross-validation set
        for(int k = 0; k < crossVal.getNumSplits(); k++){
            sumov1 = 0;
            sumov2 = 0;
            sumov3 = 0;
            sumav1 = 0;
            sumav2 = 0;
            sumav3 = 0;
            sum22 = 0;
            
            DataSetSplit dataSplit = crossVal.getValidationSet(k);
            ClassifierFactory DT1 = new ClassifierFactory(ClassifierFactory.DECISION_TREE, 1);
            ClassifierFactory DT2 = new ClassifierFactory(ClassifierFactory.DECISION_TREE, 2);
            ClassifierFactory DT3 = new ClassifierFactory(ClassifierFactory.DECISION_TREE, 3);
            ClassifierFactory DT22 = new ClassifierFactory(ClassifierFactory.DECISION_TREE, 22);
            

            //run 100 times to get avg accuracy
            for (int j = 0; j < 1; j++) {
                System.out.println("iteration");
                correct = 0.0;
                incorrect = 0.0;
                OVAClassifier OVAClassifier1 = new OVAClassifier(DT1);
                OVAClassifier OVAClassifier2 = new OVAClassifier(DT2);
                OVAClassifier OVAClassifier3 = new OVAClassifier(DT3);
                AVAClassifier avaClassifier1 = new AVAClassifier(DT1);
                AVAClassifier avaClassifier2 = new AVAClassifier(DT2);
                AVAClassifier avaClassifier3 = new AVAClassifier(DT3);
                Classifier dtClassifier22 = DT22.getClassifier();
                
                OVAClassifier1.train(dataSplit.getTrain());
                OVAClassifier2.train(dataSplit.getTrain());
                OVAClassifier3.train(dataSplit.getTrain());
                avaClassifier1.train(dataSplit.getTrain());
                avaClassifier2.train(dataSplit.getTrain());
                avaClassifier3.train(dataSplit.getTrain());
                dtClassifier22.train(dataSplit.getTrain());

                for (int i = 0; i < dataSplit.getTest().getData().size(); i++)
                    if (OVAClassifier1.classify(dataSplit.getTest().getData().get(i)) == dataSplit.getTest().getData().get(i).getLabel()) {
                        correct++;
                    } else {
                        incorrect++;
                    }
                sumov1 += correct/(correct + incorrect);
                
                for (int i = 0; i < dataSplit.getTest().getData().size(); i++)
                    if (OVAClassifier2.classify(dataSplit.getTest().getData().get(i)) == dataSplit.getTest().getData().get(i).getLabel()) {
                        correct++;
                    } else {
                        incorrect++;
                    }
                sumov2 += correct/(correct + incorrect);

                for (int i = 0; i < dataSplit.getTest().getData().size(); i++)
                    if (OVAClassifier3.classify(dataSplit.getTest().getData().get(i)) == dataSplit.getTest().getData().get(i).getLabel()) {
                        correct++;
                    } else {
                        incorrect++;
                    }
                sumov3 += correct/(correct + incorrect);

                for (int i = 0; i < dataSplit.getTest().getData().size(); i++)
                    if (avaClassifier1.classify(dataSplit.getTest().getData().get(i)) == dataSplit.getTest().getData().get(i).getLabel()) {
                        correct++;
                    } else {
                        incorrect++;
                    }
                sumav1 += correct/(correct + incorrect);
                
                for (int i = 0; i < dataSplit.getTest().getData().size(); i++)
                    if (avaClassifier2.classify(dataSplit.getTest().getData().get(i)) == dataSplit.getTest().getData().get(i).getLabel()) {
                        correct++;
                    } else {
                        incorrect++;
                    }
                sumav2 += correct/(correct + incorrect);

                for (int i = 0; i < dataSplit.getTest().getData().size(); i++)
                    if (avaClassifier3.classify(dataSplit.getTest().getData().get(i)) == dataSplit.getTest().getData().get(i).getLabel()) {
                        correct++;
                    } else {
                        incorrect++;
                    }
                sumav3 += correct/(correct + incorrect);


                for (int i = 0; i < dataSplit.getTest().getData().size(); i++)
                    if (dtClassifier22.classify(dataSplit.getTest().getData().get(i)) == dataSplit.getTest().getData().get(i).getLabel()) {
                        correct++;
                    } else {
                        incorrect++;
                    }
                sum22 += correct/(correct + incorrect);
               
            }
            System.out.println("ov1: " + sumov1/1);
            System.out.println("ov1: " + sumov2/1);
            System.out.println("ov1: " + sumov3/1);
            System.out.println("av1: " + sumav1/1);
            System.out.println("av2: " + sumav2/1);
            System.out.println("av3: " + sumav3/1);
            System.out.println("dt22: " + sum22/1);
        }
    }
}