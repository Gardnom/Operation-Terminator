using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http.Headers;
using MathNet.Numerics.LinearAlgebra;

namespace Operation_Terminator {
class Program
    {
        static void Main(string[] args) {
            NeuralNetwork nn = new NeuralNetwork(new int[]{2, 10, 10, 2});
            var input = Vector<float>.Build.DenseOfArray(new float[] {1 , 2});
            var target = Vector<float>.Build.DenseOfArray(new float[] {0, 1});
            for (int i = 0; i < 50; i++) {
                nn.Train(input, 1);
                nn.UpdateWeightsAndBiases(0.1f, 1);
            }
            //nn.Train(input, 1);
            //nn.Train(input, 1);
            
            TestMNTI();
            
            
            Console.ReadLine();
        }

        static void TestMNTI() {
            NeuralNetwork nn = new NeuralNetwork(new int[]{784, 16, 16, 10});
            float[,] pixelData = new float[40000, 784];
            int[] labels = new int[40000];
            using (var reader = new StreamReader(@"H:\dev\Operation-Terminator\Resources\mnist_train.csv")) {
               
                int i = 0;
                reader.ReadLine();
                while (!reader.EndOfStream && i < 40000) {
                    var line = reader.ReadLine();
                    var values = line.Split(",");
                    labels[i] = int.Parse(values[0]);
                    for (int k = 1; k < values.Length; k++) {
                        pixelData[i, k - 1] = int.Parse(values[k]);
                    }

                    i++;
                }
            }

            Matrix<float> inMat = Matrix<float>.Build.DenseOfArray(pixelData);
            Console.WriteLine(inMat.RowCount);
            Console.WriteLine(inMat.ColumnCount);
           
            /* var labelVec = Vector<float>.Build.Dense(10);
             labelVec[labels[0]] = 1;
             nn.Train(inMat.Row(0), labels[0]);
             */

            //nn.Train(inMat.Row(1), labels[1]);
            //nn.UpdateWeightsAndBiases(0.1f);
            
            int batchSize = 10;
            int testSampleAmount = 100;
            int trainingSampleAmount = 10000 - testSampleAmount;
            float startCost = 0.0f;
            float endCost = 0.0f;

            float cost = 0.0f;
            int epochs = 100;

            System.Random rand = new Random();
            int currEpoch = 0;
            for (int thisEpoch = 0; thisEpoch < epochs; thisEpoch++) {
                Console.WriteLine("Current epoch: " + currEpoch);
                for (int k = 0; k < trainingSampleAmount / batchSize; k++) {
                    for (int i = 0; i < batchSize; i++) {
                        var item = rand.Next(0, trainingSampleAmount - 1);
                        cost = nn.Train(inMat.Row(item), labels[item]);
                        if (k == 0 && i == 0)
                            startCost = cost;
                    
                    }
                    nn.UpdateWeightsAndBiases(0.1f, batchSize);
                }

                currEpoch++;
            }

            endCost = cost;

            //Console.WriteLine("Startcost: " + startCost);
            //Console.WriteLine("Endcost: " + endCost);

            
            
            
            int numCorrect = 0;
            for (int i = 0; i < testSampleAmount; i++) {
                var labelVec = Vector<float>.Build.Dense(10);
                labelVec[labels[trainingSampleAmount + i]] = 1;
                var outPutVec = nn.Brain(inMat.Row(trainingSampleAmount + i));
                if (labelVec.MaximumIndex() == outPutVec.MaximumIndex()) {
                    numCorrect++;
                }
            }
           
            Console.WriteLine("Number of correct guesses: " + numCorrect);
        }
    }
}
