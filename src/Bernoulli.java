// Implement a Naive Bayes classifier for text classification.

// This classifier will be used to classify fortune cookie messages into two classes:
// label messages that predict what will happen in the future as class 1
// label messages that contain a wise saying as class 0

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

/*
"""Train a Bernoulli naive Bayes classifier

        Args:
            documents (list): Each element in this list
                is a blog of text
            labels (list): The ground truth label for
                each document
        """

 */
public class Bernoulli {
    static Map<Integer, ArrayList<String>> mapClass0;
    static Map<Integer, ArrayList<String>> mapClass1;
    double[] prior;
    float[][] condprob;
    Map<String, Integer> terms;

    /*
    self._log_priors = None
        self._cond_probs = None
        self.features = None
     */
    public Bernoulli() {
        mapClass0 = new HashMap<>();
        mapClass1 = new HashMap<>();
        prior = new double[2];
        terms = new HashMap<>();
    }

/*
    """Compute log( P(Y) )
    """
    label_counts = Counter(labels)
    N = float(sum(label_counts.values()))
    self._log_priors = {k: log(v/N) for k, v in label_counts.iteritems()}

    """Feature extraction
    """
    # Extract features from each document
    X = [set(get_features(d)) for d in documents]

    # Get all features
    self.features = set([f for features in X for f in features])

    """Compute log( P(X|Y) )

       Use Laplace smoothing
       n1 + 1 / (n1 + n2 + 2)
    """
    self._cond_probs = {l: {f: 0. for f in self.features} for l in self._log_priors}

 */
    public void trainBernoulliNB() { //(C, D)
        // V <- extractVocabulary(D)
        extractVocabulary(new File("./traindata.txt"), new File("./trainlabels.txt"));
        // N <- countDocs(D)
        int N0 = mapClass0.size();
        int N1 = mapClass1.size();
        int N = N0 + N1;
        //for each c in C
        // do Nc <- countDocsInClass(D, c)
            // prior[c] <- N𝑐/N aka all the “training” documents that are of class 𝑐
        this.prior[0] = N0/N;
        this.prior[1] = N1/N;
        this.condprob = new float[terms.size()][2];

        int N0t = 0; int N1t = 0;
        // for each t in V
        for (String term: terms.keySet()) {
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
            this.condprob[terms.get(term)][0] = (N0t + 1) / (N0 + 2);
            this.condprob[terms.get(term)][1] = (N1t + 1) / (N1 + 2);
        }
        // return V, prior, condprob
    }

    public void applyBernoulliNB() { //(C, V, prior, condprob, d)
        // Vd <- extractTermsFromDoc(V, d) --- check if term in terms is in d
        // for each c in C
        // do score[c] <- log prior[c]
            // for each t in V
            // do if t is in Vd
                // To avoid number overflow, we operate on the logs of probabilities:
                // then score[c] += log condprob[t][c]
                // else score[c] += log(1 - condprob[t][c])
        // return 𝑐𝑚𝑎𝑝 = argmax𝑐inCscore[c] aka 𝑐𝑚𝑎𝑝 = argmax𝑐 𝑃(𝑐|𝑑) mapClass0: maximum a posteriori
    }

    public void extractVocabulary(File dataFile, File labelsFile) {
        Scanner scData = null;
        Scanner scLabels = null;
        try {
            scData = new Scanner(dataFile);
            scLabels = new Scanner(labelsFile);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        int index0 = 0; int index1 = 0; int indexTerms = 0;
        String line; /*int classInt;*/
        while (scData.hasNextLine() && (line = scData.nextLine()) != null) {
            //classInt = scLabels.nextInt();
            //System.out.println(line + ": "+classInt);
            String[] words = line.split("\\s+");
            if (scLabels.nextInt() == 0) {
                mapClass0.put(index0, new ArrayList<>(Arrays.asList(words)));
                for (String word : words) {
                    if(!terms.containsKey(word)) {
                        terms.put(word, indexTerms);
                        indexTerms++;
                    }
                }
                index0++;
            } else { // class is 1
                mapClass1.put(index1, new ArrayList<>(Arrays.asList(words)));
                for (String word : words) {
                    if(!terms.containsKey(word)) {
                        terms.put(word, indexTerms);
                        indexTerms++;
                    }
                }
                index1++;
            }
        }
        System.out.println("map0:");
        System.out.println(mapClass0.toString());
        System.out.println("map1:");
        System.out.println(mapClass1.toString());
    }

    public static void main(String[] args) {
        // write your code here
        Bernoulli nb = new Bernoulli();
        nb.trainBernoulliNB();
        nb.applyBernoulliNB();
    }
}
