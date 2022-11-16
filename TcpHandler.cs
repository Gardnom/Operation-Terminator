using System;
using System.Net;
using System.Net.Sockets;

namespace Operation_Terminator
{
    public class TcpHandler
    {
        private TcpListener _listener;
        private byte[] _buffer;

        public delegate void OnMessageRecieved(byte[] bytes);
        public delegate int OnGuessWanted(int imageIndex);
        
        private bool running = true;

        private OnGuessWanted _onGuessWanted;
        
        public TcpHandler(OnGuessWanted onGuessWanted) {
            _onGuessWanted = onGuessWanted;
            Int32 port = 13000;
            IPAddress localAddr = IPAddress.Parse("127.0.0.1");

            // TcpListener server = new TcpListener(port);
            _listener = new TcpListener(localAddr, port);
            _buffer = new byte[8192];
        }
        
        

        public async void Start() {
            running = true;
            _listener.Start();
            while (running) {
                Console.WriteLine("Waiting for connection...");

                TcpClient client = await _listener.AcceptTcpClientAsync();
                Console.WriteLine("Client connected!");
                HandleClient(client);
            }
        }

        private void HandleClient(TcpClient client) {
            Console.WriteLine("Handling client");
            NetworkStream stream = client.GetStream();
            byte[] buffer = new byte[1024];
            while (client.Connected) {
                stream.Read(buffer, 0, buffer.Length);
                int imageIndex = BitConverter.ToInt32(buffer, 0);
                Console.WriteLine("Image index: " + imageIndex);
                var networkGuess = _onGuessWanted(imageIndex);
                stream.Write(BitConverter.GetBytes(networkGuess));
            }

            Console.WriteLine("Client disconnected!");
        }
    }
}