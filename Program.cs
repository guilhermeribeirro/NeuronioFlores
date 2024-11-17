using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace IrisFlowerClustering
{
    class Program
    {
        static readonly string _dataPath = @"C:\Users\gui-s\Downloads\NeuronioFlores\NeuronioFlores\NeuronioFlores\iris.data.csv";

        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 0);

            IDataView dataView = mlContext.Data.LoadFromTextFile<IrisData>(_dataPath, hasHeader: false, separatorChar: ';');

            var featuresColumnName = "Features";
            var pipeline = mlContext.Transforms
                .Concatenate(featuresColumnName, nameof(IrisData.SepalLength), nameof(IrisData.SepalWidth), nameof(IrisData.PetalLength), nameof(IrisData.PetalWidth))
                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));

            var model = pipeline.Fit(dataView);

            var predictor = mlContext.Model.CreatePredictionEngine<IrisData, ClusterPrediction>(model);

            var predictions = mlContext.Data.CreateEnumerable<IrisData>(dataView, reuseRowObject: false)
                .Select(data => new IrisClusterPrediction
                {
                    Data = data,
                    Prediction = predictor.Predict(data)
                }).ToList();

            Console.WriteLine("Clusters e Classificações:");
            foreach (var prediction in predictions)
            {
                Console.WriteLine($"Dados: {prediction.Data.SepalLength}, {prediction.Data.SepalWidth}, {prediction.Data.PetalLength}, {prediction.Data.PetalWidth} => Cluster: {prediction.Prediction.PredictedClusterId}");
            }

            PlotClusters(predictions);
        }

        static void PlotClusters(List<IrisClusterPrediction> predictions)
        {
            using (var writer = new StreamWriter("ClusterResults.csv"))
            {
                writer.WriteLine("SepalLength,SepalWidth,PetalLength,PetalWidth,ClusterId");
                foreach (var prediction in predictions)
                {
                    writer.WriteLine($"{prediction.Data.SepalLength},{prediction.Data.SepalWidth},{prediction.Data.PetalLength},{prediction.Data.PetalWidth},{prediction.Prediction.PredictedClusterId}");
                }
            }

            Console.WriteLine("Resultados salvos em ClusterResults.csv. Use Python/Matplotlib para visualizar o gráfico.");
        }
    }
}