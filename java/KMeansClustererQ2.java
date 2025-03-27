import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.Scanner;
import java.util.Arrays;
import java.util.HashSet;

/**
 * KMeansClustererQ2.java - a JUnit-testable interface for the Model AI
 * Assignments k-Means Clustering exercises (Exercise 2).
 * 
 * @author Todd W. Neller
 */
public class KMeansClustererQ2 {
	private int dim; // the number of dimensions in the data
	private int k, kMin, kMax; // the allowable range of the of clusters
	private int iter; // the number of k-Means Clustering iterations per k
	private double[][] data; // the data vectors for clustering
	private double[][] centroids; // the cluster centroids
	private int[] clusters; // assigned clusters for each data point
	private Random random = new Random();

	public double[][] readData(String filename) {
		int numPoints = 0;

		try {
			Scanner in = new Scanner(new File(filename));
			try {
				dim = Integer.parseInt(in.nextLine().split(" ")[1]);
				numPoints = Integer.parseInt(in.nextLine().split(" ")[1]);
			} catch (Exception e) {
				System.err.println("Invalid data file format. Exiting.");
				e.printStackTrace();
				System.exit(1);
			}
			double[][] data = new double[numPoints][dim];
			for (int i = 0; i < numPoints; i++) {
				String line = in.nextLine();
				Scanner lineIn = new Scanner(line);
				for (int j = 0; j < dim; j++)
					data[i][j] = lineIn.nextDouble();
				lineIn.close();
			}
			in.close();
			return data;
		} catch (FileNotFoundException e) {
			System.err.println("Could not locate source file. Exiting.");
			e.printStackTrace();
			System.exit(1);
		}
		return null;
	}

	public void setData(double[][] data) {
		this.data = data;
		this.dim = data[0].length;
	}

	public double[][] getData() {
		return data;
	}

	public int getDim() {
		return dim;
	}

	public void setKRange(int kMin, int kMax) {
		this.kMin = kMin;
		this.kMax = kMax;
		this.k = kMin;
	}

	public int getK() {
		return k;
	}

	public void setIter(int iter) {
		this.iter = iter;
	}

	public double[][] getCentroids() {
		return centroids;
	}

	public int[] getClusters() {
		return clusters;
	}

	private double getDistance(double[] p1, double[] p2) {
		double sumOfSquareDiffs = 0;
		for (int i = 0; i < p1.length; i++) {
			double diff = p1[i] - p2[i];
			sumOfSquareDiffs += diff * diff;
		}
		return Math.sqrt(sumOfSquareDiffs);
	}

	public double getWCSS() {
		// Initialize WCSS (Within-Cluster Sum of Squares) to 0
		double wcss = 0.0;
		// Iterate through each data point
		for (int i = 0; i < data.length; i++) {
			// Get the cluster assignment for the current data point
			int cluster = clusters[i];
			// Calculate the Euclidean distance between the data point and its assigned
			// centroid
			double distance = getDistance(data[i], centroids[cluster]);
			// Add the squared distance to WCSS (squared to emphasize larger distances)
			wcss += distance * distance;
		}
		// Return the total WCSS for the clustering
		return wcss;
	}

	public boolean assignNewClusters() {
		// Flag to track if any cluster assignments changed
		boolean changed = false;
		// Iterate through each data point
		for (int i = 0; i < data.length; i++) {
			// Get the current data point
			double[] point = data[i];
			// Initialize variables to track the best cluster and minimum distance
			int bestCluster = -1;
			double minDistance = Double.MAX_VALUE;

			// Iterate through each centroid to find the closest one
			for (int j = 0; j < k; j++) {
				// Calculate the Euclidean distance to the current centroid
				double distance = getDistance(point, centroids[j]);
				// Update the best cluster if this distance is smaller
				if (distance < minDistance) {
					minDistance = distance;
					bestCluster = j;
				}
			}

			// If the new cluster assignment differs from the current one, update it
			if (clusters[i] != bestCluster) {
				clusters[i] = bestCluster;
				changed = true; // Mark that a change occurred
			}
		}
		// Return whether any assignments changed (used to determine convergence)
		return changed;
	}
/**
	 * Compute new centroids at the mean point of each cluster of points.
	 */
	public void computeNewCentroids() {
		// Initialize a new array to store the updated centroids
		double[][] newCentroids = new double[k][dim];
		// Set all values in newCentroids to 0
		for (int i = 0; i < k; i++) {
			Arrays.fill(newCentroids[i], 0.0);
		}
		// Array to count the number of points in each cluster
		int[] counts = new int[k];
		// Initialize counts to 0
		Arrays.fill(counts, 0);

		// Iterate through each data point to accumulate sums for each cluster
		for (int j = 0; j < data.length; j++) {
			// Get the cluster assignment for the current data point
			int cluster = clusters[j];
			// Increment the count for this cluster
			counts[cluster]++;
			// Add the data point's coordinates to the running sum for this cluster
			for (int d = 0; d < dim; d++) {
				newCentroids[cluster][d] += data[j][d];
			}
		}

		// Compute the mean for each cluster to update centroids
		for (int i = 0; i < k; i++) {
			// If the cluster has points, compute the mean
			if (counts[i] > 0) {
				for (int d = 0; d < dim; d++) {
					newCentroids[i][d] /= counts[i];
				}
			} else {
				// If the cluster is empty, retain the old centroid
				System.arraycopy(centroids[i], 0, newCentroids[i], 0, dim);
			}
		}

		// Update the centroids with the new values
		centroids = newCentroids;
	}
/**
	 * Perform k-means clustering with Forgy initialization and return the 0-based
	 * cluster assignments for corresponding data points.
	 * If iter > 1, choose the clustering that minimizes the WCSS measure.
	 * If kMin < kMax, select the k maximizing the gap statistic using 100
	 * uniform samples uniformly across given data ranges.
	 */
	public void kMeansCluster() {
		// Check if k exceeds the number of data points, which would be invalid
		if (k > data.length) {
			throw new IllegalStateException("Number of clusters (k=" + k
					+ ") cannot exceed number of data points (" + data.length + ").");
		}

		// Initialize variables to store the best clustering result
		int[] bestClusters = null;
		double[][] bestCentroids = null;
		double bestWCSS = Double.MAX_VALUE;

		// Perform iter independent runs of k-means clustering
		for (int run = 0; run < iter; run++) {
			// Forgy initialization: randomly select k data points as initial centroids
			ArrayList<Integer> indices = new ArrayList<>();
			for (int i = 0; i < data.length; i++) {
				indices.add(i);
			}
			Collections.shuffle(indices, random);
			centroids = new double[k][dim];
			for (int i = 0; i < k; i++) {
				int dataIndex = indices.get(i);
				System.arraycopy(data[dataIndex], 0, centroids[i], 0, dim);
			}

			// Initialize cluster assignments
			clusters = new int[data.length];
			assignNewClusters();

			// Perform k-means iterations until convergence
			boolean changed;
			do {
				// Update centroids based on current cluster assignments
				computeNewCentroids();
				// Reassign points to the nearest centroid
				changed = assignNewClusters();
				// Handle empty clusters by reinitializing their centroids
				int[] counts = new int[k];
				for (int cluster : clusters)
					counts[cluster]++;
				for (int i = 0; i < k; i++) {
					if (counts[i] == 0) {
						int randomIdx = random.nextInt(data.length);
						System.arraycopy(data[randomIdx], 0, centroids[i], 0, dim);
						changed = true; // Force another iteration to reassign points
					}
				}
			} while (changed);

			// Compute WCSS for the current clustering
			double currentWCSS = getWCSS();
			// Update the best clustering if the current WCSS is lower
			if (currentWCSS < bestWCSS) {
				bestWCSS = currentWCSS;
				bestClusters = clusters.clone();
				bestCentroids = new double[k][dim];
				for (int i = 0; i < k; i++) {
					System.arraycopy(centroids[i], 0, bestCentroids[i], 0, dim);
				}
			}
		}

		// Set the final clustering to the best result
		clusters = bestClusters;
		centroids = bestCentroids;

		// Safety check to ensure a valid clustering was produced
		if (clusters == null || centroids == null) {
			throw new IllegalStateException("kMeansCluster failed to produce a valid clustering.");
		}
	}

	/**
	 * Export cluster data in the given data output format to the file provided.
	 * 
	 * @param filename the destination file
	 */
	public void writeClusterData(String filename) {
		try {
			FileWriter out = new FileWriter(filename);

			out.write(String.format("%% %d dimensions\n", dim));
			out.write(String.format("%% %d points\n", data.length));
			out.write(String.format("%% %d clusters/centroids\n", k));
			out.write(String.format("%% %f within-cluster sum of squares\n", getWCSS()));
			for (int i = 0; i < k; i++) {
				out.write(i + " ");
				for (int j = 0; j < dim; j++)
					out.write(centroids[i][j] + (j < dim - 1 ? " " : "\n"));
			}
			for (int i = 0; i < data.length; i++) {
				out.write(clusters[i] + " ");
				for (int j = 0; j < dim; j++)
					out.write(data[i][j] + (j < dim - 1 ? " " : "\n"));
			}
			out.flush();
			out.close();
		} catch (Exception e) {
			System.err.println("Error writing to file");
			e.printStackTrace();
			System.exit(1);
		}
	}

	/**
	 * Read UNIX-style command line parameters to as to specify the type of k-Means
	 * Clustering algorithm applied to the formatted data specified.
	 * "-k int" specifies both the minimum and maximum number of clusters. "-kmin
	 * int" specifies the minimum number of clusters. "-kmax int" specifies the
	 * maximum number of clusters.
	 * "-iter int" specifies the number of times k-Means Clustering is performed in
	 * iteration to find a lower local minimum.
	 * "-in filename" specifies the source file for input data. "-out filename"
	 * specifies the destination file for cluster data.
	 * 
	 * @param args command-line parameters specifying the type of k-Means Clustering
	 */
	public static void main(String[] args) {
		int kMin = 2, kMax = 2, iter = 1;
		ArrayList<String> attributes = new ArrayList<String>();
		ArrayList<Integer> values = new ArrayList<Integer>();
		int i = 0;
		String infile = null;
		String outfile = null;
		while (i < args.length) {
			if (args[i].equals("-k") || args[i].equals("-kmin") || args[i].equals("-kmax")
					|| args[i].equals("-iter")) {
				attributes.add(args[i++].substring(1));
				if (i == args.length) {
					System.err.println("No integer value for"
							+ attributes.get(attributes.size() - 1) + ".");
					System.exit(1);
				}
				try {
					values.add(Integer.parseInt(args[i]));
					i++;
				} catch (Exception e) {
					System.err.printf("Error parsing \"%s\" as an integer.", args[i]);
					System.exit(2);
				}
			} else if (args[i].equals("-in")) {
				i++;
				if (i == args.length) {
					System.err.println("No string value provided for input source");
					System.exit(1);
				}
				infile = args[i];
				i++;
			} else if (args[i].equals("-out")) {
				i++;
				if (i == args.length) {
					System.err.println("No string value provided for output source");
					System.exit(1);
				}
				outfile = args[i];
				i++;
			}
		}

		for (i = 0; i < attributes.size(); i++) {
			String attribute = attributes.get(i);
			if (attribute.equals("k"))
				kMin = kMax = values.get(i);
			else if (attribute.equals("kmin"))
				kMin = values.get(i);
			else if (attribute.equals("kmax"))
				kMax = values.get(i);
			else if (attribute.equals("iter"))
				iter = values.get(i);
		}

		KMeansClustererQ2 km = new KMeansClustererQ2();
		km.setKRange(kMin, kMax);
		km.setIter(iter);
		km.setData(km.readData(infile));
		km.kMeansCluster();
		km.writeClusterData(outfile);
	}
}