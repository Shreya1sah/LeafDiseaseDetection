package com.example.smartleafai;

import java.util.Arrays;
public class EnsembleFuzzyFusion {
    public static int fuzzyRankBasedFusion(float[] denseNetPred, float[] inceptionPred, float[] xceptionPred) {
        int numClasses = denseNetPred.length;
        float[] fusionScores = new float[numClasses];

        float[][] predictions = { denseNetPred, inceptionPred, xceptionPred };

        for (float[] pred : predictions) {
            int[] ranks = getRanks(pred);
            for (int i = 0; i < numClasses; i++) {
                fusionScores[i] += 1.0f / (ranks[i] + 1); // +1 to avoid division by 0
            }
        }

        int maxIndex = 0;
        float maxScore = fusionScores[0];
        for (int i = 1; i < fusionScores.length; i++) {
            if (fusionScores[i] > maxScore) {
                maxScore = fusionScores[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private static int[] getRanks(float[] prediction) {
        int n = prediction.length;
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        Arrays.sort(indices, (a, b) -> Float.compare(prediction[b], prediction[a]));

        int[] ranks = new int[n];
        for (int rank = 0; rank < n; rank++) {
            ranks[indices[rank]] = rank;
        }
        return ranks;
    }

}
