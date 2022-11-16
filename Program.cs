using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http.Headers;
using MathNet.Numerics.LinearAlgebra;

namespace Operation_Terminator {
class Program
{
        static void Main(string[] args) {

            //var nn = NeuralNetwork.LoadModelFromFile(@"H:\dev\Operation-Terminator\Resources\model.model");
            //Console.WriteLine("Loaded model with " + nn.m_PercentCorrect + " percent accuracy");
            //TestMNISTModel(nn);

            /*var (data, labels) = GetDataMNIST(@"H:\dev\Operation-Terminator\Resources\mnist_test.csv");
            
            
            TcpHandler tcpHandler = new TcpHandler(imageIndex => {
                var output = nn.Brain(data.Row(imageIndex));
                return output.MaximumIndex();
            });
            tcpHandler.Start();
            */
            
            int numModelsWanted = 100;
            
            for (int i = 0; i < numModelsWanted; i++) {
                TestMNTI();
            }
            
            
            Console.ReadLine();
        }

        static void TestDummyNetwork() {
            NeuralNetwork nn = new NeuralNetwork(new int[]{2, 10, 10, 2});
            var input = Vector<float>.Build.DenseOfArray(new float[] {1 , 2});
            var target = Vector<float>.Build.DenseOfArray(new float[] {0, 1});
            for (int i = 0; i < 50; i++) {
                nn.Train(input, 1);
                nn.UpdateWeightsAndBiases(0.1f, 1);
            }
            //nn.Train(input, 1);
            //nn.Train(input, 1);

        }

        static (Matrix<float>, int[]) GetDataMNIST(string path) {
            float[,] pixelData = new float[60000, 784];
            int[] labels = new int[60000];
            using (var reader = new StreamReader(path)) {
               
                int i = 0;
                reader.ReadLine();
                while (!reader.EndOfStream && i < 60000) {
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
            //Console.WriteLine(inMat.RowCount);
            //Console.WriteLine(inMat.ColumnCount);

            return (inMat, labels);
        }

        static void TrainMNIST(NeuralNetwork nn, int numEpochs = 100) {
            var (inMat, labels) = GetDataMNIST(@"H:\dev\Operation-Terminator\Resources\mnist_train.csv");

            int batchSize = 10;
            int testSampleAmount = 100;
            int trainingSampleAmount = 10000 - testSampleAmount;
            float startCost = 0.0f;
            float endCost = 0.0f;

            float cost = 0.0f;

            System.Random rand = new Random();
            int currEpoch = 0;
            for (int epoch = 0; epoch < numEpochs; epoch++) {
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
        }

        static void TestMNISTModel(NeuralNetwork nn) {
            int testSampleAmount = 10000;
            int numCorrect = 0;
            var (inMat, labels) = GetDataMNIST(@"H:\dev\Operation-Terminator\Resources\mnist_test.csv");
            for (int i = 0; i < testSampleAmount; i++) {
                var labelVec = Vector<float>.Build.Dense(10);
                labelVec[labels[i]] = 1;
                var outPutVec = nn.Brain(inMat.Row(i));
                if (labelVec.MaximumIndex() == outPutVec.MaximumIndex()) {
                    numCorrect++;
                }
            }
            
            Console.WriteLine("Number of correct guesses: " + numCorrect);
        }
        
        static void TestMNTI() {
            NeuralNetwork nn = new NeuralNetwork(new int[]{784, 16, 16, 10});
           
            int testSampleAmount = 10000;
            int numCorrect = 0;
            
            TrainMNIST(nn);
            
            var (inMat, labels) = GetDataMNIST(@"H:\dev\Operation-Terminator\Resources\mnist_test.csv");
            for (int i = 0; i < testSampleAmount; i++) {
                var labelVec = Vector<float>.Build.Dense(10);
                labelVec[labels[i]] = 1;
                var outPutVec = nn.Brain(inMat.Row(i));
                if (labelVec.MaximumIndex() == outPutVec.MaximumIndex()) {
                    numCorrect++;
                }
            }
            
            
            Console.WriteLine("Number of correct guesses: " + numCorrect);
            if (numCorrect >= 7000) {
                float percentCorrect = (((float) numCorrect / testSampleAmount) * 100);
                Console.WriteLine("Percent correct: " + percentCorrect);
                Console.WriteLine("Should save model!");
                nn.SaveModelToFile(@$"H:\dev\Operation-Terminator\Resources\model{String.Format("{0:0.00}", percentCorrect)}%.model", percentCorrect);
            }
        }
    }
}
