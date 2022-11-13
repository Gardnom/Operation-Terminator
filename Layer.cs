using System;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace Operation_Terminator
{
    public class Layer
    {
        public int m_NNodes, m_NInputs;
        public Vector<float> nodes;
        public Vector<float> biases;
        public Matrix<float> weights;

        public Layer(int nInputs, int nNodes) {
            m_NInputs = nInputs;
            m_NNodes = nNodes;

            nodes = Vector<float>.Build.Dense(nNodes);
            biases = Vector<float>.Build.Dense(nNodes);
            weights = Matrix<float>.Build.Random(nInputs, nNodes);
        }

        public void ForwardPass(Vector<float> inputs) {
            if (inputs.Count != m_NInputs) {
                Console.WriteLine("Wrong number of inputs in forward pass");
                return;
            }
            nodes = inputs * weights;
            
        }
        
        public Matrix<float> ForwardPassBatch(Matrix<float> inputs) {
            /*if (inputs.Count != m_NInputs) {
                Console.WriteLine("Wrong number of inputs in forward pass");
                return;
            }*/
            return inputs * weights;
            
        }

        

        public void ActivationReLU() {
            nodes.MapInplace(val => {
                if (val < 0) return 0;
                return val;
            });
        }
    }
}