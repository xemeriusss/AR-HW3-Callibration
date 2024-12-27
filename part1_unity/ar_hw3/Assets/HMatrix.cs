using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Optimization;
using UnityEngine;

public class HomographyCalculator : MonoBehaviour
{

    public static double[,] CalculateHomography(double[,] scene, double[,] image)
    {
        if (scene.GetLength(0) != image.GetLength(0) || scene.GetLength(1) != 2 || image.GetLength(1) != 2)
        {
            throw new ArgumentException("Scene and image points must have the same length.");
        }

        int numPoints = scene.GetLength(0);

        // Normalize points to improve numerical stability
        double[,] T_scene = NormalizePoints(scene, out double[,] sceneNormalized);
        double[,] T_image = NormalizePoints(image, out double[,] imageNormalized);

        // Compute initial H using DLT
        double[,] H_init = ComputeDLT(sceneNormalized, imageNormalized);

        // Perform nonlinear optimization
        double[] optimizedH = OptimizeHomographyGradientDescent(H_init, sceneNormalized, imageNormalized);

        // Denormalize the optimized H to get the final result
        double[,] H = new double[3, 3];
        for (int i = 0; i < 9; i++)
        {
            H[i / 3, i % 3] = optimizedH[i];
        }

        H = MatrixMultiply(MatrixInverse(T_image), MatrixMultiply(H, T_scene));

        return H;
    }
    
    private static double[] OptimizeHomographyGradientDescent(double[,] initialH, double[,] scene, double[,] image, double learningRate = 0.01, int maxIterations = 5000)
    {
        int numPoints = scene.GetLength(0);

        // Flatten the initial H matrix to a 1D array
        double[] paramsH = new double[9];
        for (int i = 0; i < 9; i++)
        {
            paramsH[i] = initialH[i / 3, i % 3];
        }

        // Gradient descent loop
        for (int iter = 0; iter < maxIterations; iter++)
        {
            double[] gradients = new double[9];

            for (int i = 0; i < numPoints; i++)
            {
                double x = scene[i, 0]; // Scene point
                double y = scene[i, 1]; // Scene point
                double X = image[i, 0]; // Image point
                double Y = image[i, 1]; // Image point
                
                double wx = paramsH[6] * x + paramsH[7] * y + paramsH[8]; // Denominator for perspective division
                double hx = (paramsH[0] * x + paramsH[1] * y + paramsH[2]) / wx; // Projected x
                double hy = (paramsH[3] * x + paramsH[4] * y + paramsH[5]) / wx; // Projected y

                double dx = hx - X; // Error in x
                double dy = hy - Y; // Error in y

                for (int j = 0; j < 9; j++)
                {
                    // Compute partial derivative of the loss function
                    double derivative = ComputePartialDerivative(j, x, y, paramsH, dx, dy, wx);

                    // Accumulate gradients
                    gradients[j] += derivative;
                }
            }

            // Update parameters using gradients
            for (int j = 0; j < 9; j++)
            {
                paramsH[j] -= learningRate * gradients[j];
            }
        }

        return paramsH; // Return the optimized parameters
    }

    private static double ComputePartialDerivative(int j, double x, double y, double[] paramsH, double dx, double dy, double wx)
    {
        double derivative = 0;

        if (j < 3)
        {
            derivative = dx * x / wx + dy * y / wx; // Derivative for H[0, 0], H[0, 1], H[0, 2]
        }
        else if (j < 6)
        {
            derivative = dx * x / wx + dy * y / wx; // Derivative for H[1, 0], H[1, 1], H[1, 2]
        }
        else
        {
            derivative = -dx * (paramsH[j - 6] * x + paramsH[j - 5] * y) / Math.Pow(wx, 2); // Derivative for H[2, 0], H[2, 1], H[2, 2]
        }

        return derivative; // Return the computed derivative
    }

    // To compute initial homography using DLT
    private static double[,] ComputeDLT(double[,] scene, double[,] image)
    {
        int numPoints = scene.GetLength(0);
        var A = DenseMatrix.OfArray(new double[2 * numPoints, 9]);

        // Construct the matrix A 
        for (int i = 0; i < numPoints; i++)
        {
            double x = scene[i, 0];
            double y = scene[i, 1];
            double X = image[i, 0];
            double Y = image[i, 1];
            
            A[2 * i, 0] = -x; 
            A[2 * i, 1] = -y;
            A[2 * i, 2] = -1;
            A[2 * i, 6] = x * X;
            A[2 * i, 7] = y * X;
            A[2 * i, 8] = X;

            A[2 * i + 1, 3] = -x;
            A[2 * i + 1, 4] = -y;
            A[2 * i + 1, 5] = -1;
            A[2 * i + 1, 6] = x * Y;
            A[2 * i + 1, 7] = y * Y;
            A[2 * i + 1, 8] = Y;
        }

        // Perform SVD 
        var svd = A.Svd();

        // Extract the last column of V as the solution
        var V = svd.VT.Transpose();
        double[,] H = new double[3, 3];
        for (int i = 0; i < 9; i++)
        {
            H[i / 3, i % 3] = V[i, V.ColumnCount - 1];
        }

        return H;
    }

    private static double[,] NormalizePoints(double[,] points, out double[,] normalizedPoints)
    {
        int numPoints = points.GetLength(0);
        double meanX = 0, meanY = 0;

        for (int i = 0; i < numPoints; i++)
        {
            meanX += points[i, 0];
            meanY += points[i, 1];
        }

        // Compute the mean of the points
        meanX /= numPoints;
        meanY /= numPoints;

        double scale = 0;

        // Compute the scale factor
        for (int i = 0; i < numPoints; i++)
        {
            scale += Math.Sqrt(Math.Pow(points[i, 0] - meanX, 2) + Math.Pow(points[i, 1] - meanY, 2));
        }
        scale = Math.Sqrt(2) / (scale / numPoints);

        // Compute the transformation matrix
        double[,] T = new double[3, 3]
        {
            { scale, 0, -scale * meanX },
            { 0, scale, -scale * meanY },
            { 0, 0, 1 }
        };

        normalizedPoints = new double[numPoints, 2];

        // Normalize the points using the transformation matrix
        for (int i = 0; i < numPoints; i++)
        {
            double x = points[i, 0];
            double y = points[i, 1];
            normalizedPoints[i, 0] = T[0, 0] * x + T[0, 1] * y + T[0, 2];
            normalizedPoints[i, 1] = T[1, 0] * x + T[1, 1] * y + T[1, 2];
        }

        return T;
    }

    private static double[,] MatrixMultiply(double[,] A, double[,] B)
    {
        var matrixA = DenseMatrix.OfArray(A);
        var matrixB = DenseMatrix.OfArray(B);
        var result = matrixA * matrixB;
        return result.ToArray();
    }

    private static double[,] MatrixInverse(double[,] A)
    {
        var matrixA = DenseMatrix.OfArray(A);
        var result = matrixA.Inverse();
        return result.ToArray();
    }

    public static double CalculateError(double[,] scene, double[,] image, double[,] H)
    {
        double totalError = 0.0;
        int numPoints = scene.GetLength(0);
        
        for (int i = 0; i < numPoints; i++)
        {
            // Project the scene point onto the image plane using the homography
            double[] projected = ProjectPoint(new double[] { scene[i, 0], scene[i, 1] }, H);

            // Euclidean distance between the projected and actual image points
            double error = Math.Sqrt(Math.Pow(projected[0] - image[i, 0], 2) + Math.Pow(projected[1] - image[i, 1], 2));

            // Accumulate the error
            totalError += error;
        }

        return totalError / numPoints; // Return the average error
    }

    public static double CalculatePointError(double[] originalPoint, double[] projectedPoint)
    {
        // Euclidean distance between the projected and actual image points
        return Math.Sqrt(Math.Pow(projectedPoint[0] - originalPoint[0], 2) + Math.Pow(projectedPoint[1] - originalPoint[1], 2));
    }

    private static double[] ProjectPoint(double[] point, double[,] H)
    {
        double x = point[0];
        double y = point[1];
        double z = H[2, 0] * x + H[2, 1] * y + H[2, 2]; // Get z coordinate for perspective division

        double u = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / z; // Projected x
        double v = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / z; // Projected y

        return new double[] { u, v }; // Return the projected point
    }

    public static double[] ProjectSceneToImage(double[] scenePoint, double[,] H)
    {
        // Project a scene point (S1, S2, S3) onto the image plane
        return ProjectPoint(scenePoint, H);
    }

    // public static double[] ProjectImageToScene(double[] imagePoint, double[,] H)
    // {
    //     // Compute the inverse of the homography matrix
    //     double[,] HInverse = MatrixInverse(H);

    //     // Project an image point (I1, I2, I3) back onto the scene plane
    //     return ProjectPoint(imagePoint, HInverse);
    // }


    // RANSAC implementation ==================================================

    public static double[,] CalculateHomographyRANSAC(double[,] scenePoints, double[,] imagePoints, double[,] confidenceMatrix, int maxIterations = 1000, double threshold = 5.0)
    {
        if (scenePoints.GetLength(0) != confidenceMatrix.GetLength(0) || imagePoints.GetLength(0) != confidenceMatrix.GetLength(1))
        {
            throw new ArgumentException("Dimensions of confidence matrix must match the number of marker and image points.");
        }

        int numscenePoints = scenePoints.GetLength(0);
        int numImagePoints = imagePoints.GetLength(0);

        System.Random rand = new System.Random();
        double[,] bestHomography = null;
        int bestInlierCount = 0;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Select four correspondences using the confidence matrix
            List<Tuple<int, int>> sampledPairs = SampleMatches(confidenceMatrix, numscenePoints, numImagePoints, rand);

            if (sampledPairs.Count < 4) continue; // Skip invalid sampling

            // Extract source and destination points for homography estimation
            double[,] src = new double[4, 2];
            double[,] dst = new double[4, 2];
            for (int i = 0; i < 4; i++)
            {
                src[i, 0] = scenePoints[sampledPairs[i].Item1, 0];
                src[i, 1] = scenePoints[sampledPairs[i].Item1, 1];
                dst[i, 0] = imagePoints[sampledPairs[i].Item2, 0];
                dst[i, 1] = imagePoints[sampledPairs[i].Item2, 1];
            }

            // Compute the homography for this subset
            double[,] H = HomographyCalculator.CalculateHomography(src, dst);

            // Count inliers based on the reprojection error
            int inlierCount = CountInliers(scenePoints, imagePoints, H, threshold);

            // Update best homography if the current one is better
            if (inlierCount > bestInlierCount)
            {
                bestInlierCount = inlierCount;
                bestHomography = H;
            }
        }

        if (bestHomography == null)
        {
            throw new InvalidOperationException("Failed to compute a valid homography.");
        }

        Debug.Log($"RANSAC: Best homography with {bestInlierCount} inliers.");

        return bestHomography;
    }

    private static List<Tuple<int, int>> SampleMatches(double[,] confidenceMatrix, int numscenePoints, int numImagePoints, System.Random rand)
    {
        List<Tuple<int, int>> pairs = new List<Tuple<int, int>>();

        while (pairs.Count < 4)
        {
            // Sample based on confidence matrix probabilities
            int sceneIdx = rand.Next(0, numscenePoints);
            double[] confidenceRow = new double[numImagePoints];
            for (int j = 0; j < numImagePoints; j++)
                confidenceRow[j] = confidenceMatrix[sceneIdx, j];

            int imageIdx = WeightedRandomChoice(confidenceRow, rand);

            if (!pairs.Contains(Tuple.Create(sceneIdx, imageIdx)))
            {
                pairs.Add(Tuple.Create(sceneIdx, imageIdx));
            }
        }

        return pairs;
    }

    private static int WeightedRandomChoice(double[] probabilities, System.Random rand)
    {
        double total = 0;
        foreach (var p in probabilities)
            total += p;

        double randomValue = rand.NextDouble() * total;
        for (int i = 0; i < probabilities.Length; i++)
        {
            randomValue -= probabilities[i];
            if (randomValue <= 0)
            {
                return i;
            }
        }

        return probabilities.Length - 1; // Fallback
    }

    private static int CountInliers(double[,] scenePoints, double[,] imagePoints, double[,] H, double threshold)
    {
        int inliers = 0;
        int numscenePoints = scenePoints.GetLength(0);

        for (int i = 0; i < numscenePoints; i++)
        {
            double[] scenePoint = { scenePoints[i, 0], scenePoints[i, 1] };
            double[] projectedImagePoint = HomographyCalculator.ProjectPoint(scenePoint, H);

            double error = Math.Sqrt(
                Math.Pow(projectedImagePoint[0] - imagePoints[i, 0], 2) +
                Math.Pow(projectedImagePoint[1] - imagePoints[i, 1], 2)
            );

            if (error < threshold)
            {
                inliers++;
            }
        }

        return inliers;
    }

    void Start()
    {   
        // Found the corner of checkerboard in the image manually
        // Image1
        double[,] imagePoints = new double[4, 2]
        {
            { 475, 514 },  // top-left
            { 2096, 532 }, // top-right
            { 454, 2612 }, // bottom-left
            { 2075, 2624 } // bottom-right
        };

        // Image2
        double[,] imagePoints2 = new double[4, 2]
        {
            { 276, 496 }, // top-left
            { 1954, 439 }, // top-right
            { 249, 2610 }, // bottom-left
            { 1936, 2692 } // bottom-right
        };

        // Image3
        double[,] imagePoints3 = new double[4, 2]
        {
            { 501, 593 }, // top-left
            { 2160, 721 }, // top-right
            { 490, 2683 }, // bottom-left
            { 1960, 2624 } // bottom-right
        };

        double[,] scenePoints = new double[4, 2]
        {
            { 0, 0 }, // top-left
            { 0, 700 }, // top-right
            { 900, 0 }, // bottom-left
            { 900, 700 } // bottom-right
        };

        // 1.1 Calculate homography matrix
        double[,] H = CalculateHomography(scenePoints, imagePoints);
        double[,] H2 = CalculateHomography(scenePoints, imagePoints2);
        double[,] H3 = CalculateHomography(scenePoints, imagePoints3);

        Debug.Log($"Homography Matrix - Image1:\n{H[0, 0]} {H[0, 1]} {H[0, 2]}\n{H[1, 0]} {H[1, 1]} {H[1, 2]}\n{H[2, 0]} {H[2, 1]} {H[2, 2]}");

        Debug.Log($"Homography Matrix - Image2:\n{H2[0, 0]} {H2[0, 1]} {H2[0, 2]}\n{H2[1, 0]} {H2[1, 1]} {H2[1, 2]}\n{H2[2, 0]} {H2[2, 1]} {H2[2, 2]}");

        Debug.Log($"Homography Matrix - Image3:\n{H3[0, 0]} {H3[0, 1]} {H3[0, 2]}\n{H3[1, 0]} {H3[1, 1]} {H3[1, 2]}\n{H3[2, 0]} {H3[2, 1]} {H3[2, 2]}");

        // Error calculation of the homography matrix
        double error = CalculateError(scenePoints, imagePoints, H);
        double error2 = CalculateError(scenePoints, imagePoints2, H2);
        double error3 = CalculateError(scenePoints, imagePoints3, H3);
        Debug.Log($"Average Error 1: {error}");
        Debug.Log($"Average Error 2: {error2}");
        Debug.Log($"Average Error 3: {error3}");


        // 1.5 Five-point correspondences
        double[,] additionalImgPoints = new double[3, 2]
        {
            { 1285,1573 }, // Center of the image
            { 1264,2618 }, 
            { 1285,523 }
        };

        double[,] additionalImgPoints2 = new double[3, 2]
        {
            { 1105, 1553 }, // near center
            { 1100, 2651 }, // near bottom edge
            { 1115, 468 }   // near top edge
        };

        double[,] additionalImgPoints3 = new double[3, 2]
        {
            { 1230, 1638 }, // near center
            { 1225, 2653 }, // near bottom edge
            { 1330, 657 }   // near top edge
        };

        double[,] additionalScenePoints = new double[3, 2]
        {
            { 450,350 }, 
            { 900,350 }, 
            { 0,350 }    
        };

        // Scene points to project onto the image
        Debug.Log("Additional Point Matches:");
        for (int i = 0; i < 3; i++)
        {
            double[] projectedPoint = ProjectPoint(new double[] { additionalScenePoints[i, 0], additionalScenePoints[i, 1] }, H);
            Debug.Log($"1- Scene Point {additionalScenePoints[i, 0]}, {additionalScenePoints[i, 1]} -> Image Point {projectedPoint[0]:F2}, {projectedPoint[1]:F2}");

            double errorPoint = CalculatePointError(new double[] { additionalImgPoints[i, 0], additionalImgPoints[i, 1] }, projectedPoint);
            Debug.Log($"Error: {errorPoint}");

            double[] projectedPoint2 = ProjectPoint(new double[] { additionalScenePoints[i, 0], additionalScenePoints[i, 1] }, H2);
            Debug.Log($"2- Scene Point {additionalScenePoints[i, 0]}, {additionalScenePoints[i, 1]} -> Image Point {projectedPoint2[0]:F2}, {projectedPoint2[1]:F2}");

            double errorPoint2 = CalculatePointError(new double[] { additionalImgPoints2[i, 0], additionalImgPoints2[i, 1] }, projectedPoint2);
            Debug.Log($"Error: {errorPoint2}");

            double[] projectedPoint3 = ProjectPoint(new double[] { additionalScenePoints[i, 0], additionalScenePoints[i, 1] }, H3);
            Debug.Log($"3- Scene Point {additionalScenePoints[i, 0]}, {additionalScenePoints[i, 1]} -> Image Point {projectedPoint3[0]:F2}, {projectedPoint3[1]:F2}");

            double errorPoint3 = CalculatePointError(new double[] { additionalImgPoints3[i, 0], additionalImgPoints3[i, 1] }, projectedPoint3);
            Debug.Log($"Error: {errorPoint3}");
        }

        // Sample points from pdf

        // 1.6 Scene points to project onto the image
        double[,] scenePointsSample = new double[3, 2]
        {
            { 7.5, 5.5 },
            { 6.3, 3.3 },
            { 0.1, 0.1 }
        };

        // Scene points to project onto the image
        Debug.Log("Additional Point Matches from PDF:");
        for (int i = 0; i < 3; i++)
        {
            double[] projectedPointSample = ProjectPoint(new double[] { scenePointsSample[i, 0], scenePointsSample[i, 1] }, H);
            Debug.Log($"1- Scene Point {scenePointsSample[i, 0]}, {scenePointsSample[i, 1]} -> Image Point {projectedPointSample[0]:F2}, {projectedPointSample[1]:F2}");

            double[] projectedPointSample2 = ProjectPoint(new double[] { scenePointsSample[i, 0], scenePointsSample[i, 1] }, H2);
            Debug.Log($"2- Scene Point {scenePointsSample[i, 0]}, {scenePointsSample[i, 1]} -> Image Point {projectedPointSample2[0]:F2}, {projectedPointSample2[1]:F2}");

            double[] projectedPointSample3 = ProjectPoint(new double[] { scenePointsSample[i, 0], scenePointsSample[i, 1] }, H3);
            Debug.Log($"3- Scene Point {scenePointsSample[i, 0]}, {scenePointsSample[i, 1]} -> Image Point {projectedPointSample3[0]:F2}, {projectedPointSample3[1]:F2}");
        }

        // 1.7 Image points to project back onto the scene
        double[,] imagePointsSample = new double[3, 2]
        {
            { 500, 400 },
            { 86, 167 },
            { 10, 10 }
        };

        // Calculate the inverse of the homography matrices
        double[,] HInverse = MatrixInverse(H);
        double[,] HInverse2 = MatrixInverse(H2);
        double[,] HInverse3 = MatrixInverse(H3);

        // Image points to project back to the scene
        Debug.Log("Inverse Projection (Image to Scene):");
        for (int i = 0; i < 3; i++)
        {
            double[] projectedScenePointSample = ProjectPoint(new double[] { imagePointsSample[i, 0], imagePointsSample[i, 1] }, HInverse);
            Debug.Log($"1- Image Point {imagePointsSample[i, 0]}, {imagePointsSample[i, 1]} -> Scene Point {projectedScenePointSample[0]:F2}, {projectedScenePointSample[1]:F2}");

            double[] projectedScenePointSample2 = ProjectPoint(new double[] { imagePointsSample[i, 0], imagePointsSample[i, 1] }, HInverse2);
            Debug.Log($"2- Image Point {imagePointsSample[i, 0]}, {imagePointsSample[i, 1]} -> Scene Point {projectedScenePointSample2[0]:F2}, {projectedScenePointSample2[1]:F2}");

            double[] projectedScenePointSample3 = ProjectPoint(new double[] { imagePointsSample[i, 0], imagePointsSample[i, 1] }, HInverse3);
            Debug.Log($"3- Image Point {imagePointsSample[i, 0]}, {imagePointsSample[i, 1]} -> Scene Point {projectedScenePointSample3[0]:F2}, {projectedScenePointSample3[1]:F2}");
        }

        // 1.2 RANSAC
        double[,] scenePointsRansac = new double[,]
        {
            { 0, 0 },     // Top-left
            { 0, 700 },   // Top-right
            { 900, 0 },   // Bottom-left
            { 900, 700 }, // Bottom-right
            { 450, 350 }  // Center
        };

        // Image1
        double[,] imagePointsRansac = new double[,]
        {
            { 475, 514 },   // Top-left
            { 2096, 532 },  // Top-right
            { 454, 2612 },  // Bottom-left
            { 2075, 2624 }, // Bottom-right
            { 1265, 1585 }  // Center
        };

        // Confidence matrix that I found manually
        double[,] confidenceMatrix = new double[,]
        {
            { 0.9, 0.1, 0.05, 0.02, 0.03 }, // Confidence for point 0
            { 0.1, 0.85, 0.03, 0.02, 0.05 }, // Confidence for point 1
            { 0.05, 0.02, 0.9, 0.03, 0.04 }, // Confidence for point 2
            { 0.02, 0.03, 0.04, 0.95, 0.03 }, // Confidence for point 3
            { 0.03, 0.05, 0.04, 0.03, 0.85 }  // Confidence for point 4
        };

        double[,] homographyMatrixRansac = CalculateHomographyRANSAC(scenePointsRansac, imagePointsRansac, confidenceMatrix);

        Debug.Log("Homography Matrix (RANSAC):");
        for (int i = 0; i < 3; i++)
        {
            Debug.Log($"{homographyMatrixRansac[i, 0]} {homographyMatrixRansac[i, 1]} {homographyMatrixRansac[i, 2]}");
        }

        double errorRansac = CalculateError(scenePointsRansac, imagePointsRansac, homographyMatrixRansac);
        Debug.Log($"Average Error (RANSAC): {errorRansac}");

        // RANSAC with additional points
        Debug.Log("RANSAC with additional points:");
        for (int i = 0; i < 3; i++)
        {
            double[] projectedPoint = ProjectPoint(new double[] { additionalScenePoints[i, 0], additionalScenePoints[i, 1] }, homographyMatrixRansac);
            Debug.Log($"1- Scene Point {additionalScenePoints[i, 0]}, {additionalScenePoints[i, 1]} -> Image Point {projectedPoint[0]:F2}, {projectedPoint[1]:F2}");

            double errorPoint = CalculatePointError(new double[] { additionalImgPoints[i, 0], additionalImgPoints[i, 1] }, projectedPoint);
            Debug.Log($"Error: {errorPoint}");
        }


    }
}




