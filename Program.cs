using System;
using MathNet.Numerics.LinearAlgebra;

namespace Operation_Terminator {
class Program
    {
        static void Main(string[] args) {
            NeuralNetwork nn = new NeuralNetwork();
            
            while (true) {
                Console.Write("Give a number to the machine!: ");
                string? input = Console.ReadLine();
                if(input == null){
                    break;
                }
                if(input == "q") break;
                float result;
                if (!float.TryParse(input, out result)) {
                    Console.WriteLine("You must give it a number!");
                    continue;    
                }

                float[,] r = {{0}, {1}, {2}};
                int[] labels = {2, 0, 1};
                var brainInput = Matrix<float>.Build.DenseOfArray(r);
             
                var resMat = nn.BrainBatch(brainInput);
                var normResMat = NeuralNetwork.SoftMax(resMat);
                var losses = nn.TrainBatch(brainInput, labels);
                Console.WriteLine("Losses: "+ losses);
                Console.WriteLine("The machine gives you back: " + normResMat[0, 0]);
                //Console.WriteLine("Normalized: " + normResMat[0, 0]);
            }
            
            
        }
    }
}
