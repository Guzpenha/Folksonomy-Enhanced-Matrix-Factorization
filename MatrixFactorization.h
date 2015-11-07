/*
  @author: guzpenha@dcc.ufmg.br
*/

#include <unordered_map>
#include <vector>
#include <math.h>
#include <dlib/matrix.h>
#include "jsoncons/json.hpp"
/*
Class variables:
---------------
		R 		: matrix of ratings, dimension N x M
		U 		: matrix of user latent factors, dimension N x K
		V 		: matrix of item latent factors, dimension K x M
		K 		: number of latent features
		M     : number of itens
		N 		: number of users
		
		alpha	: learning rate
		beta	:	regularization parameter
		steps	: number of steps for gradient descent operations
		learning_tolerance: the minimum gain acepted in each step (stops if gain < tolerance). 
		user_bias: the bias of each user
		item_bias: the bias of each item
		alpha_bias_item: the learning rate of the item bias.
		alpha_bias_user: the learning rate of the user bias. 
		beta_bias_item: regularization parameter for item bias.
		beta_bias_user: regularization parameter for user bias.

		overall_mean: mean of all ratings.
		overall_deviation: deviation of all ratings.

		testSet: test set of the ratings.
		trainSet: train set of the ratings.

Functions:
----------
		modelLatentVariablesBA(): factorize matrix R, into U and V. (Bias Aware Matrix Factorization)
		predictAllRatingsBA(): returns the prediction matrix U x V. (predict BAMF)
		predictRatingsBA(): predicts ratings for target tuples <user,item>.
		calculateRMSE(): calculate RMSE over two vectors of predictions and actual ratings.
		normalizePrediction(): normalize prediction with algorithm of choice.

		calculate_bias(): calculates the user and item bias over all ratings.
*/
		
using namespace std;
using jsoncons::json;
class MatrixFactorization {
	public:
		explicit MatrixFactorization(int s,int k, unordered_map<int, unordered_map<int, double> >  ratings, int qntItens,int qntUser, string jsonName, unordered_map<string,int> itemStringToID);
		virtual ~MatrixFactorization();

		// FEMF
		void modelLatentVariablesFEMF();

		// BAMF
		void modelLatentVariablesBADlib(bool useTrainSet);
		vector<double> predictRatingsBADlib(vector<int> users, vector<int> itens);


		double calculateRMSE(vector<double> P, vector<double> actual);
		vector<double> normalizePrediction(vector<double> predictions, int type);															

	private:
		void calculate_bias();
		void content_analyzer();

		string jsonFileName;
		vector<json> itemContent;
		unordered_map<string,int>  itemStringToID;
		unordered_map<string,int>  docsWithTerm;
		vector< unordered_map<string,double> > itemTermWeights;
		vector< unordered_map<int,int> > itemInvertedIndex;
		
		unordered_map<int, unordered_map<int, double> >  R; 
		unordered_map<int, vector<double> >  U; 
		unordered_map<int, vector<double> >  V; 
		vector<double> item_bias;
		vector<double> user_bias;


		int M;
		int N;
		int K; 
		int steps;
		double alpha; 
		double beta;
		double alpha_bias_item; 
		double alpha_bias_user; 
		double beta_bias_item;
		double beta_bias_user;
		
		double overall_mean;
		double overall_deviation;
		double learning_tolerance;

		unordered_map<int, unordered_map<int, double> >  testSet; 
		unordered_map<int, unordered_map<int, double> >  trainSet; 
		dlib::matrix <double> U_matrix;
		dlib::matrix <double> V_matrix;
};

