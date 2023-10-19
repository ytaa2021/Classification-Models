package ml.classifiers;
import ml.DataSet;
import ml.Example;
import java.util.Set;
import java.util.ArrayList;
import java.util.HashMap;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.util.List;
import java.io.IOException;

//import javafx.scene.chart.PieChart.Data;

/*
 * Printing getData returns examples: [1.0 0:0 1:0 2:0 3:0 4:1 5:0] Label, feature index:feature value
 */
public class DecisionTreeClassifier implements Classifier {
    public int maxDepth;
    public String currPath = "";
    public HashMap<Integer, String> originalFeatures;
    public  DecisionTreeNode tree;

    /*
    * @param data
    */
    public double[] errorCount(DataSet data) {
        int n = data.getData().size();
        int[] zeroSurvive = new int[n];
        int[] zeroDied = new int[n];
        int[] oneSurvive = new int[n];
        int[] oneDied = new int[n];
        for (int i = 0; i < n; i++) {
            Example currEx = data.getData().get(i);
            Double label = currEx.getLabel();
            int m = currEx.getFeatureSet().size();
            for (int j = 0; j < m; j++) {
                if (currEx.getFeature(j) == 0 && label == -1) {
                    zeroDied[j]++;
                }
                if (currEx.getFeature(j) == 1 && label == -1) {
                    oneDied[j]++;
                }
                if (currEx.getFeature(j) == 0 && label == 1) {
                    zeroSurvive[j]++;
                }
                if (currEx.getFeature(j) == 1 && label == 1) {
                    oneSurvive[j]++;
                }
            }
        }
        return accuracyCalc(zeroSurvive, oneSurvive, zeroDied, oneDied, data);
    }

    public double[] accuracyCalc(int[] zeroSurvive, int[] oneSurvive, int[] zeroDied, int[] oneDied, DataSet data) {
        double[] accuracy = new double[zeroDied.length];
        int majority0;
        int majority1;
        
        for (int i = 0; i < oneSurvive.length; i++) {
            //Zeros
            majority0 = Math.max(zeroSurvive[i], zeroDied[i]);
            majority1 = Math.max(oneSurvive[i], oneDied[i]);
            accuracy[i] = ((double) (majority0 + majority1)) / ((double) data.getData().size());
        }
        
        return accuracy;
    }

    
    public void train(DataSet data) {
        DecisionTreeNode trainTree = trainRecurse(data, 0,0);
        this.tree = trainTree;
    }



    public DecisionTreeNode trainRecurse(DataSet data, int depth, int fileNumber) {
        int posCount = 0;
        int negCount = 0;
        for (int i = 0; i < data.getData().size(); i++) {
            if (data.getData().get(i).getLabel() == -1) {
                negCount++;
            } else {
                posCount++;
            }
        }
        double parentPrediction;
        if (posCount > negCount) {
            parentPrediction = 1;
        } else {
            parentPrediction = -1;
        }
        


        //if theere is only one feature left = BC
        double[] accuracy = errorCount(data);
        for(int i = 0; i < accuracy.length; i++) {
            //System.out.print(accuracy[i] + " ");
        }
        //System.out.println();
        double ourMax = 0.0;
        int ourMaxIndex = 0;
        //System.out.println("new loop");
        for (int i = 0; i < accuracy.length; i++){
            //System.out.println(accuracy[i]);
            //System.out.println(ourMax);
            if (accuracy[i] > ourMax) {
                ourMax = accuracy[i];
                ourMaxIndex = i;
            }
        }

        String ourMaxFeature = data.getFeatureMap().get(ourMaxIndex);
        int originalIndex = -1;  // Initialize to an invalid value
        if (fileNumber == 0) {
            originalFeatures = data.getFeatureMap();
        }
        for (HashMap.Entry<Integer, String> entry : originalFeatures.entrySet()) {
            if (entry.getValue().equals(ourMaxFeature)) {
                originalIndex = entry.getKey();
                break;
            }
        }
        //create currnode
        
        DecisionTreeNode currNode = new DecisionTreeNode(originalIndex);
        //split datasets on this feature 0/1
        //if side has different labels call train on dataset without this feature for only that value of that feature
        String inputFilePath = "";
        if (fileNumber == 0) {
            inputFilePath = "data/titanic-train.csv";
        } else {
            inputFilePath = currPath;
        }
        fileNumber++;
        String outputFilePath0 = "data/titanic-train" + fileNumber + "0.csv";
        String outputFilePath1 = "data/titanic-train" + fileNumber + "1.csv";
        String columnNameToRemove = data.getFeatureMap().get(ourMaxIndex);

        
        splitDataSet(inputFilePath, outputFilePath0, outputFilePath1, columnNameToRemove);
        try (
        BufferedReader in0 = new BufferedReader(new FileReader(outputFilePath0));
        BufferedReader in1 = new BufferedReader(new FileReader(outputFilePath1));
        ) {

        String line0 = in0.readLine();
        String line1 = in1.readLine();
        if (line0 == null || line0.isEmpty()){
            Example currEx;
            Double label;
            double zeroDied = 0;
            double zeroSurvived = 0;
            for (int i = 0; i < data.getData().size(); i++) {
                currEx = data.getData().get(i);
                label = currEx.getLabel();
                if (currEx.getFeature(ourMaxIndex) == 0 && label == -1) {
                    zeroDied++;
                }
                if (currEx.getFeature(ourMaxIndex) == 0 && label == 1) {
                    zeroSurvived++;
                }
            }
            if (zeroSurvived > zeroDied) {
                DecisionTreeNode leftLeaf = new DecisionTreeNode(1.0);
                currNode.setLeft(leftLeaf);
            } else {
                DecisionTreeNode leftLeaf = new DecisionTreeNode(-1.0);
                currNode.setLeft(leftLeaf);
            }
        } else {
            DataSet zeroesData = new DataSet(outputFilePath0);
            //Left
            //If the feature we are on always predicts same
            //if side always has same label then build a leaf
            Boolean isLeaf = true;
            Double initialLabelValue = zeroesData.getData().get(0).getLabel();
            for (int i = 1; i < zeroesData.getData().size(); i++){
                if (zeroesData.getData().get(i).getLabel() != initialLabelValue) {
                    isLeaf = false;
                }
            }
            ArrayList<Example> dataArr = zeroesData.getData();
            Boolean allSame = true;
            int labelDied = 0;
            int labelSurvived = 0;
            double currLabel;
            for (int j = 0; j < dataArr.get(0).getFeatureSet().size(); j++) {
                for (int i = 0; i < dataArr.size() - 1; i++){
                    currLabel = dataArr.get(i).getLabel();
                    double featureSet0 = dataArr.get(i).getFeature(j);
                    double featureSet1 = dataArr.get(i+1).getFeature(j);
                    if (featureSet0 != featureSet1){
                        allSame = false;
                    } 
                }
            }

            if (isLeaf) {
                DecisionTreeNode leftNode = new DecisionTreeNode(initialLabelValue);
                currNode.setLeft(leftNode);
            //if there is only one feature
            } else if (allSame) {
                for (int i =0; i < dataArr.size(); i++) {
                    currLabel = dataArr.get(i).getLabel();
                    if (currLabel == -1.0) {
                        labelDied++;
                    }
                    if (currLabel == 1.0) {
                        labelSurvived++;
                    }
                }
                //System.out.println("l");
                if (labelDied > labelSurvived) {
                    DecisionTreeNode leftNode = new DecisionTreeNode(-1.0);
                    currNode.setLeft(leftNode);
                } else if (labelDied < labelSurvived) {
                    DecisionTreeNode leftNode = new DecisionTreeNode(1.0);
                    currNode.setLeft(leftNode);
                } else {
                    DecisionTreeNode leftNode = new DecisionTreeNode(parentPrediction);
                    currNode.setLeft(leftNode);
                }
                
            } else 
            if (zeroesData.getFeatureMap().size() == 1){
                Example currEx;
                Double label;
                double zeroDied = 0;
                double zeroSurvived = 0;
                double oneDied = 0;
                double oneSurvived = 0;
                for (int i = 0; i < zeroesData.getData().size(); i++) {
                    currEx = data.getData().get(i);
                    label = currEx.getLabel();
                    if (currEx.getFeature(ourMaxIndex) == 0 && label == -1) {
                        zeroDied++;
                    }
                    if (currEx.getFeature(ourMaxIndex) == 1 && label == -1) {
                        oneDied++;
                    }
                    if (currEx.getFeature(ourMaxIndex) == 0 && label == 1) {
                        zeroSurvived++;
                    }
                    if (currEx.getFeature(ourMaxIndex) == 1 && label == 1) {
                        oneSurvived++;
                    }
                }
                if (zeroSurvived > zeroDied) {
                    DecisionTreeNode leftLeaf = new DecisionTreeNode(1.0);
                    currNode.setLeft(leftLeaf);
                } else if (zeroSurvived < zeroDied) {
                    DecisionTreeNode leftLeaf = new DecisionTreeNode(-1.0);
                    currNode.setLeft(leftLeaf);
                } else {
                    DecisionTreeNode leftLeaf = new DecisionTreeNode(parentPrediction);
                    currNode.setLeft(leftLeaf);
                }

                if (oneSurvived > oneDied) {
                    DecisionTreeNode rightLeaf = new DecisionTreeNode(1.0);
                    currNode.setRight(rightLeaf);
                } else if (oneSurvived < oneDied){
                    DecisionTreeNode rightLeaf = new DecisionTreeNode(-1.0);
                    currNode.setRight(rightLeaf);
                } else {
                    DecisionTreeNode rightLeaf = new DecisionTreeNode(parentPrediction);
                    currNode.setRight(rightLeaf);
                }
                
            } else {
                currPath = outputFilePath0;
                if (depth == maxDepth - 1 || maxDepth == 0) {
                    Example currEx;
                    Double label;
                    double zeroDied = 0;
                    double zeroSurvived = 0;
                    for (int i = 0; i < zeroesData.getData().size(); i++) {
                        currEx = data.getData().get(i);
                        label = currEx.getLabel();
                        if (currEx.getFeature(ourMaxIndex) == 0 && label == -1) {
                            zeroDied++;
                        }
                        if (currEx.getFeature(ourMaxIndex) == 0 && label == 1) {
                            zeroSurvived++;
                        }
                    }
                    if (zeroSurvived > zeroDied) {
                        DecisionTreeNode leftLeaf = new DecisionTreeNode(1.0);
                        currNode.setLeft(leftLeaf);
                    } else if (zeroSurvived < zeroDied){
                        DecisionTreeNode leftLeaf = new DecisionTreeNode(-1.0);
                        currNode.setLeft(leftLeaf);
                    } else {
                        DecisionTreeNode rightLeaf = new DecisionTreeNode(parentPrediction);
                        currNode.setLeft(rightLeaf);
                    }

                } else {
                    DecisionTreeNode leftNode = trainRecurse(zeroesData, depth + 1, fileNumber);
                    currNode.setLeft(leftNode);
                }
                
            }
        }
        
        if (line1 == null || line1.isEmpty()){
            Example currEx;
            Double label;
            double zeroDied = 0;
            double zeroSurvived = 0;
            for (int i = 0; i < data.getData().size(); i++) {
                currEx = data.getData().get(i);
                label = currEx.getLabel();
                if (currEx.getFeature(ourMaxIndex) == 0 && label == -1) {
                    zeroDied++;
                }
                if (currEx.getFeature(ourMaxIndex) == 0 && label == 1) {
                    zeroSurvived++;
                }
            }
            if (zeroSurvived > zeroDied) {
                DecisionTreeNode rightLeaf = new DecisionTreeNode(1.0);
                currNode.setRight(rightLeaf);
            } else {
                DecisionTreeNode rightLeaf = new DecisionTreeNode(-1.0);
                currNode.setRight(rightLeaf);
            }
        } else {
            DataSet onesData = new DataSet(outputFilePath1);
                //Right
            //if label values are all same on one side BC help
            boolean isLeaf = true;
            double initialLabelValue = onesData.getData().get(0).getLabel();
            for (int i = 1; i < onesData.getData().size(); i++){
                if (onesData.getData().get(i).getLabel() != initialLabelValue) {
                    isLeaf = false;
                }
            }
            ArrayList<Example> dataArr = onesData.getData();
            boolean allSame = true;
            int labelDied = 0;
            int labelSurvived = 0;
            double currLabel;
            for (int j = 0; j < dataArr.get(0).getFeatureSet().size(); j++) {
                for (int i = 0; i < dataArr.size() - 1; i++){
                    currLabel = dataArr.get(i).getLabel();
                    double featureSet0 = dataArr.get(i).getFeature(j);
                    double featureSet1 = dataArr.get(i+1).getFeature(j);
                    if (featureSet0 != featureSet1){
                        allSame = false;
                    } 
                }
            }

            //if label values are all same on one side BC
            if (isLeaf) {
                DecisionTreeNode rightNode = new DecisionTreeNode(initialLabelValue);
                currNode.setRight(rightNode);
            } else if (allSame) {
                for (int i =0; i < dataArr.size(); i++) {
                    currLabel = dataArr.get(i).getLabel();
                    if (currLabel == -1.0) {
                        labelDied++;
                    }
                    if (currLabel == 1.0) {
                        labelSurvived++;
                    }
                }
                if (labelDied > labelSurvived) {
                    DecisionTreeNode rightNode = new DecisionTreeNode(-1.0);
                    currNode.setRight(rightNode);
                } else if (labelDied < labelSurvived) {
                    DecisionTreeNode rightNode = new DecisionTreeNode(1.0);
                    currNode.setRight(rightNode);
                } else {
                    DecisionTreeNode rightNode = new DecisionTreeNode(parentPrediction);
                    currNode.setRight(rightNode);
                }
            }
            else if (onesData.getFeatureMap().size() == 1){
                Example currEx;
                Double label;
                double zeroDied = 0;
                double zeroSurvived = 0;
                double oneDied = 0;
                double oneSurvived = 0;
                for (int i = 0; i < onesData.getData().size(); i++) {
                    currEx = data.getData().get(i);
                    label = currEx.getLabel();
                    if (currEx.getFeature(ourMaxIndex) == 0 && label == -1) {
                        zeroDied++;
                    }
                    if (currEx.getFeature(ourMaxIndex) == 1 && label == -1) {
                        oneDied++;
                    }
                    if (currEx.getFeature(ourMaxIndex) == 0 && label == 1) {
                        zeroSurvived++;
                    }
                    if (currEx.getFeature(ourMaxIndex) == 1 && label == 1) {
                        oneSurvived++;
                    }
                }
                if (zeroSurvived > zeroDied) {
                    DecisionTreeNode leftLeaf = new DecisionTreeNode(1.0);
                    currNode.setLeft(leftLeaf);
                } else if (zeroSurvived < zeroDied){
                    DecisionTreeNode leftLeaf = new DecisionTreeNode(-1.0);
                    currNode.setLeft(leftLeaf);
                } else {
                    DecisionTreeNode leftLeaf = new DecisionTreeNode(parentPrediction);
                    currNode.setLeft(leftLeaf);
                }

                if (oneSurvived > oneDied) {
                    DecisionTreeNode rightLeaf = new DecisionTreeNode(1.0);
                    currNode.setRight(rightLeaf);
                } else if (oneSurvived < oneDied) {
                    DecisionTreeNode rightLeaf = new DecisionTreeNode(-1.0);
                    currNode.setRight(rightLeaf);
                } else {
                    DecisionTreeNode rightLeaf = new DecisionTreeNode(parentPrediction);
                    currNode.setRight(rightLeaf);
                }
            } else {
                currPath = outputFilePath1;
                if (depth == maxDepth - 1 || maxDepth == 0) {
                    Example currEx;
                    Double label;
                    double oneDied = 0;
                    double oneSurvived = 0;
                    for (int i = 0; i < onesData.getData().size(); i++) {
                        currEx = data.getData().get(i);
                        label = currEx.getLabel();
                        if (currEx.getFeature(ourMaxIndex) == 0 && label == -1) {
                            oneDied++;
                        }
                        if (currEx.getFeature(ourMaxIndex) == 0 && label == 1) {
                            oneSurvived++;
                        }
                    }
                    if (oneSurvived > oneDied) {
                        DecisionTreeNode rightLeaf = new DecisionTreeNode(1.0);
                        currNode.setRight(rightLeaf);
                    } else if (oneSurvived < oneDied){
                        DecisionTreeNode rightLeaf = new DecisionTreeNode(parentPrediction);
                        currNode.setRight(rightLeaf);
                    }

                } else {
                DecisionTreeNode rightNode = trainRecurse(onesData, depth + 1, fileNumber);
                currNode.setRight(rightNode);
                }
            }
        }
    

        //
        return currNode;
        }
        catch (IOException e) {
            e.printStackTrace();
            return currNode;
        }
        /*
         * look at each feature individually
         */
    }


    public static void splitDataSet(String inputcsv, String outputcsv0, String outputcsv1, String coltoremove) {
        try (BufferedReader in = new BufferedReader(new FileReader(inputcsv));
         BufferedWriter out1 = new BufferedWriter(new FileWriter(outputcsv1));
         BufferedWriter out0 = new BufferedWriter(new FileWriter(outputcsv0))) {

        String line = in.readLine();
        
        // Skip lines that start with #
        while (line != null && line.startsWith("#")) {
            line = in.readLine();
        }

        if (line != null) {
            // Find the index of the column to remove
            String[] headers = line.split(",");
            int indexToRemove = -1;
            for (int i = 0; i < headers.length; i++) {
                if (headers[i].equals(coltoremove)) {
                    indexToRemove = i;
                    break;
                }
            }

            if (indexToRemove == -1) {
                System.out.println("Column name not found.");
                return;
            }

            // Remove the column from the header
            List<String> modifiedHeaders = new ArrayList<>();
            for (int i = 0; i < headers.length; i++) {
                if (i != indexToRemove) {
                    modifiedHeaders.add(headers[i]);
                }
            }

            String headerStr = String.join(",", modifiedHeaders);

            out1.write(headerStr);
            out1.newLine();

            out0.write(headerStr);
            out0.newLine();

            // Process each subsequent line
            while ((line = in.readLine()) != null) {
                String[] values = line.split(",");
                List<String> modifiedValues = new ArrayList<>();
                for (int i = 0; i < values.length; i++) {
                    if (i != indexToRemove) {
                        modifiedValues.add(values[i]);
                    }
                }

                if (values[indexToRemove].equals("1")) {
                    out1.write(String.join(",", modifiedValues));
                    out1.newLine();
                } else {
                    out0.write(String.join(",", modifiedValues));
                    out0.newLine();
                }
            }
        }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
   
   /**
    * Classify the example.  Should only be called *after* train has been called.
    * 
    * @param example
    * @return the class label predicted by the classifier for this example
    */
    public double classify(Example example) {
        //go through the tree: if 1 then right, if 0 then left
        return classifyRecurse(example, tree);
    }
    
    public double classifyRecurse(Example example, DecisionTreeNode tree){
        if (tree.isLeaf()){ return tree.prediction(); }
        else {
            if (example.getFeature(tree.getFeatureIndex()) == 0.0){
                return classifyRecurse(example, tree.getLeft());
            } else {
                return classifyRecurse(example, tree.getRight());
            }
        }
    }
    public String toString() {
        return tree.treeString();
    }
    public DecisionTreeClassifier() {
        maxDepth = -2;
    }

    public void setDepthLimit(int depth) {
        maxDepth = depth;
    }

    public static void main(String[] args) {
        DecisionTreeClassifier classifier = new DecisionTreeClassifier();
        classifier.setDepthLimit(1);
        // Now you can call other methods on the classifier instance
        // For example, set the depth limit
        //classifier.setDepthLimit(5);
        DataSet data = new DataSet("data/titanic-train.csv");
        classifier.train(data);

        // TODO: Add training and classification steps here
    }
}

