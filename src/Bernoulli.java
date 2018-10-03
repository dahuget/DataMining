/*
 * Dana Huget, V00860786 & Ricardo Rodriguez, V00797811
 * October 2, 2018
 * SENG474-A01 Assignment 1 Question 4
 *
 * Implementation of a Naive Bayes Bernoulli model text classification
 *
 * This classifier is used to classify fortune cookie messages into two classes:
 * Class 1: label messages that predict what will happen in the future
 * Class 0: label messages that contain a wise saying
 *
 */

import java.io.*;
import java.util.*;

public class Bernoulli {
    static Map<Integer, ArrayList<String>> mapClass0 = new HashMap<>();
    static Map<Integer, ArrayList<String>> mapClass1 = new HashMap<>();
    static double[] prior = new double[2];
    static float[][] condprob;
    static Map<String, Integer> V = new HashMap<>();

    public static void trainBernoulliNB(File dataFile, File labelsFile) { //(C, D)
        // V <- extractVocabulary(D)
        extractVocabulary(dataFile, labelsFile);
        // N <- countDocs(D)
        float N0 = mapClass0.size();
        float N1 = mapClass1.size();
        float N = N0 + N1;
        //for each c in C
        // do Nc <- countDocsInClass(D, c)
        // prior[c] <- N𝑐/N aka all the “training” documents that are of class 𝑐
        prior[0] = N0/N;
        prior[1] = N1/N;
        condprob = new float[V.size()][2];

        // for each t in V
        for (String term: V.keySet()) {
            int N0t = 0; int N1t = 0;
            // do Nct <- countDocsInClassContainingTerm(D, 𝑐, t)
            for(int i = 0; i < N0; i++) {
                if(mapClass0.get(i).contains(term)) {
                    N0t++;
                }
            }
            for(int i = 0; i < N1; i++) {
                if(mapClass1.get(i).contains(term)) {
                    N1t++;
                }
            }
            // To avoid the zero frequency problem we do:
            // condprob[t][c] <- (N𝑐t + 1) / (N𝑐 + 2) aka the fraction of documents of class 𝑐 that contain term t
            condprob[V.get(term)][0] = (N0t + 1) / (N0 + 2);
            condprob[V.get(term)][1] = (N1t + 1) / (N1 + 2);
        }
        // return V, prior, condprob
    }

    public static Map<ArrayList<String>, Integer> getDocs(File data, File labels) {
        Map<ArrayList<String>, Integer> docs = new HashMap<>();
        Scanner scData = null;
        Scanner scLabels = null;
        try {
            scData = new Scanner(data);
            scLabels = new Scanner(labels);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        String line;
        while (scData.hasNextLine() && (line = scData.nextLine()) != null) {
            String[] words = line.split("\\s+");
            docs.put(new ArrayList<>(Arrays.asList(words)), scLabels.nextInt());
        }
        return docs;
    }

    public static ArrayList<String> extractTermsFromDoc(ArrayList<String> doc) {
        ArrayList<String> Vd = new ArrayList<>();
        for (String word : doc) {
            if (!Vd.contains(word)) {
                Vd.add(word);
            }
        }
        return Vd;
    }

    public static int applyBernoulliNB(ArrayList<String> doc) { //(C, V, prior, condprob, d)
        // Vd <- extractTermsFromDoc(V, d)
        ArrayList<String> Vd = extractTermsFromDoc(doc);
        // for each c in C ie 0 and 1
        // do score[c] <- log prior[c]
        double score[] = new double[2];
        score[0] = Math.log(prior[0]);
        score[1] = Math.log(prior[1]);
        // for each t in V
        for (String term : V.keySet()) {
            // do if t is in Vd
            if (Vd.contains(term)) {
                // To avoid number overflow, we operate on the logs of probabilities:
                // then score[c] += log condprob[t][c]
                score[0] += Math.log(condprob[V.get(term)][0]);
                score[1] += Math.log(condprob[V.get(term)][1]);
            } else {
                // else score[c] += log(1 - condprob[t][c])
                score[0] += Math.log(1 - condprob[V.get(term)][0]);
                score[1] += Math.log(1 - condprob[V.get(term)][1]);
            }
        }
        // The class with the highest log probability score is the most probable
        // 𝑚𝑎𝑝 = argmax𝑐inCscore[c] aka 𝑐𝑚𝑎𝑝 = arg max𝑐 𝑃(𝑐|𝑑) maximum a posteriori
        if (score[0] > score[1]) {
            return 0;
        }
        return 1;
    }

    public static void extractVocabulary(File dataFile, File labelsFile) {
        Map<ArrayList<String>, Integer> docs = getDocs(dataFile, labelsFile);
        int index0 = 0; int index1 = 0; int indexTerms = 0;
        for (Map.Entry<ArrayList<String>, Integer> doc : docs.entrySet()) {
            for (String word : doc.getKey()) {
                if(!V.containsKey(word)) {
                    V.put(word, indexTerms);
                    indexTerms++;
                }
            }
            if (doc.getValue().intValue() == 0) {
                mapClass0.put(index0, doc.getKey());
                index0++;
            } else {
                mapClass1.put(index1, doc.getKey());
                index1++;
            }
        }
    }

    public static void writeResults(float trainAccuracy, float testAccuracy) {
        try (FileWriter writer = new FileWriter("results.txt")) {
            writer.write("Percent Accuracy Report\n\n");
            writer.write("training data: traindata.txt & trainlabels.txt\n");
            writer.write("testing data: traindata.txt\n");
            writer.write("accuracy: "+trainAccuracy+"\n\n");
            writer.write("training data: traindata.txt & trainlabels.txt\n");
            writer.write("testing data: testdata.txt\n");
            writer.write("testing data: traindata.txt\n");
            writer.write("accuracy: "+testAccuracy+"\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        trainBernoulliNB(new File("./traindata.txt"), new File("./trainlabels.txt"));
        Map<ArrayList<String>, Integer> trainDocs = getDocs(new File("./traindata.txt"), new File("./trainlabels.txt"));
        float error = 0; float total = trainDocs.size();
        for (Map.Entry<ArrayList<String>, Integer> doc : trainDocs.entrySet())  {
            int label = applyBernoulliNB(doc.getKey());
            if (label != doc.getValue()) {
                error++;
            }
        }
        float trainPercent = ((total-error)/total)*100;

        Map<ArrayList<String>, Integer> testDocs = getDocs(new File("./testdata.txt"), new File("./testlabels.txt"));
        error = 0; total = testDocs.size();
        for (Map.Entry<ArrayList<String>, Integer> doc : testDocs.entrySet())  {
            int label = applyBernoulliNB(doc.getKey());
            if (label != doc.getValue()) {
                error++;
            }
        }
        float testPercent = ((total-error)/total)*100;

        writeResults(trainPercent, testPercent);
    }
}
