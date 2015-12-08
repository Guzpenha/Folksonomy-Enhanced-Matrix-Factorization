/*
  @author: guzpenha@dcc.ufmg.br
*/

#include "MatrixFactorization.h"
#include <time.h>
#include <typeinfo>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <sstream>
#include <dlib/matrix.h>
#include "jsoncons/json.hpp"
#include "prettyprint.hpp"

#define PVF(X) for(int x=0;x<X.size();x++){printf("%f ",X[x]);}printf("\n");
#define SZ(X) ((int)(X).size())
#define REP(I, N) for (int I = 0; I < (N); ++I)
#define REPP(I, A, B) for (int I = (A); I < (B); ++I)
#define REPM(I, N) for(auto I = (N.begin()); I != N.end();I++)
#define MP make_pair
#define PB push_back
#define F first
#define S second
#define MIN_RATING 1.0
#define MAX_RATING 10.0

#define CEILMAX 1 
#define RANGE  2
#define ZSCORE 3


using namespace std;
using jsoncons::json;
using jsoncons::json_exception;    
//--------------------------------------------------------------------------------------//
//    => 	Constructor and destructor																										//
//																																											//
//--------------------------------------------------------------------------------------//
MatrixFactorization::MatrixFactorization(int s,int k,unordered_map<int, unordered_map<int,double> >  r, int qntItens, int qntUser, string jsonFile, unordered_map<string,int> isbnToID){
	steps = s;
  alpha = 0.002;
  beta = 0.5;
  beta_tags = 0.1;
	K=k;
	alpha_bias_item = 0.001; 
	alpha_bias_user = 0.001; 
	beta_bias_item = 0.5;
	beta_bias_user = 0.5;
	R = r;	
	N = qntUser;
	M = qntItens;
	user_bias.resize(N,0);
	item_bias.resize(M,0);
	learning_tolerance=0.01;	
	jsonFileName = jsonFile;
	itemStringToID = isbnToID;
}
MatrixFactorization::~MatrixFactorization() {}


//--------------------------------------------------------------------------------------//
//    => 	PREDICTION AND MODELING FUNCTIONS																													//
//																																											//
//--------------------------------------------------------------------------------------//

/*******************************************************************\
MatrixFactorization::normalizePredictions()

	parameters:
		=> predictions , a vector of doubles of predictions made.
		=> type , int containing the type of normalization to be used.
				 CEILMAX(TRIM) =  1 
				 RANGE   = 2
				 ZSCORE = 3
	return:
		=> predictions, vector of doubles with predictions normalized.
/********************************************************************/
vector<double> MatrixFactorization::normalizePrediction(vector<double> predictions, int type){
	// cout<<"Normalizing predictions."<<endl;
	int typeNormalization = type;
	
	if(typeNormalization == CEILMAX){
		REP(i,SZ(predictions)){			
			predictions[i] = ( (predictions[i] > MAX_RATING)? MAX_RATING : predictions[i] );
			predictions[i] = ( (predictions[i] < MIN_RATING)? MIN_RATING : predictions[i] );		
		}
	}else if (typeNormalization == RANGE){
		double maxValue=1;
		double minValue=10;

		REP(i,SZ(predictions)){
			maxValue = ( (predictions[i] > maxValue)? predictions[i]  : maxValue );
			minValue = ( (predictions[i] < minValue)? predictions[i]  : minValue );
		}
		REP(i,SZ(predictions)){
			predictions[i] = (((double)(predictions[i] - minValue)/ (double)(maxValue- minValue)) * 10.0);
			predictions[i] = ( (predictions[i] > MAX_RATING)? MAX_RATING : predictions[i] );
			predictions[i] = ( (predictions[i] < MIN_RATING)? MIN_RATING : predictions[i] );		
		}
	}else if(typeNormalization == ZSCORE){
		double mean,deviaton;
		mean = deviaton = 0;	
		int count =0;
		REP(i,SZ(predictions)){
			mean +=predictions[i];			
			count++;
		}
		mean = (mean/(double)count);	
		REP(i,SZ(predictions)){						
			deviaton += pow((predictions[i] - mean) , 2 );							
		}
		deviaton = deviaton/(double)count;
		deviaton = sqrt(deviaton);
		REP(i,SZ(predictions)){							
			predictions[i] = ((predictions[i] - mean)/deviaton);
			predictions[i] = overall_mean +  (predictions[i] * overall_deviation);				
			predictions[i] = ( (predictions[i] > MAX_RATING)? MAX_RATING : predictions[i] );
			predictions[i] = ( (predictions[i] < MIN_RATING)? MIN_RATING : predictions[i] );											
		}
	}
	return predictions;
}

//--------------------/
//  BIAS AWARE MF
//--------------------/

/*******************************************************************\
MatrixFactorization::predictRatingsBADlib()

	parameters:
		=> users, vector containing the user of tuple (i) to be predicted
		=> itens, vector containing the item of tuple (i) to be predicted

	return:
		=> P, a vector containing predictions for tuples <user[i],item[i]>
/********************************************************************/
vector<double> MatrixFactorization::predictRatingsBADlib(vector<int> users, vector<int> itens){
	// cout<<"Predicting target ratings."<<endl;
	vector<double>  P;
	REP(i,SZ(users)){
		dlib::matrix<double> A,B;
		A = dlib::rowm(U_matrix,users[i]);
		B = dlib::colm(V_matrix,itens[i]);				
		double dotProduct = dlib::dot(A,B);
		P.PB(overall_mean + user_bias[users[i]]+ item_bias[itens[i]] + dotProduct) ;
		if(P.back() != P.back()){
			P.pop_back();
			P.PB(overall_mean);
		}
	}
	return P;
}

/*******************************************************************\
MatrixFactorization::modelLatentVariablesBADlib()

	parameters:
		=> none, R must have been provided when object was initialized.

	return:
		=> none, class variables U and V containing latent features are 
		calculated in place, along with bias_user and bias_item.
/********************************************************************/
void MatrixFactorization::modelLatentVariablesBADlib(bool useTrainSet){
	cout<<"Starting gradient descent bias aware DLIB."<<endl;
	int totalN = SZ(R);
	int totalM = M;
	if(useTrainSet)
		R =trainSet;

	calculate_bias();

	// Initializes latent matrixes with random value.
	U_matrix = dlib::randm(totalN,K);
	V_matrix = dlib::randm(K,totalM);	
	REP(i,totalN){
		REP(j,K){
			U_matrix(i,j)=0.0001;
		}
	}
	REP(i,K){
		REP(j,totalM){
			V_matrix(i,j)=0.0001;
		}
	}

	// Move to gradient of function for each prediction made
	double previous_error = 10000000;
	REP(step,steps){			
		// cout<<U_matrix<<endl;
		// cout<<V_matrix<<endl;
		REPM(it,R){
			REPM(jt,(*it).S){
				int i = (*it).F; int j = (*jt).F;								
				dlib::matrix<double> A,B;
				A = dlib::rowm(U_matrix,i);
				B = dlib::colm(V_matrix,j);				
				double dotProduct = dlib::dot(A,B);
				// cout<<dotProduct<<endl;				
				double errorij = R[i][j] - (overall_mean +  user_bias[i] + item_bias[j] + dotProduct);				

				REP(k,K){				
					double gradientU = (errorij * V_matrix(k,j) - beta * U_matrix(i,k));
					double gradientV = (errorij * U_matrix(i,k) - beta * V_matrix(k,j));					
					U_matrix(i,k) = U_matrix(i,k) + alpha * (gradientU); 
					V_matrix(k,j) = V_matrix(k,j) + alpha * (gradientV);
					user_bias[i] = user_bias[i] + alpha_bias_user * ( errorij - (beta_bias_user * user_bias[i]));
					item_bias[j] = item_bias[j] + alpha_bias_item * ( errorij - (beta_bias_item * item_bias[j]));
				}
			}			
		}				
		// cout<<"Step : "<<step<<endl;		
	}
}

/*******************************************************************\
MatrixFactorization::calculate_bias()

	parameters:
		=> none, R must have been provided when object was initialized.

	return:
		=> none, item_bias, user_bias and means are returned in 
		 class variables
/********************************************************************/
void MatrixFactorization::calculate_bias(){
	//Calculate overall mean
	double ratings_mean = 0;
	double ratings_count = 0;
	double deviation = 0;
	REP(i,N)
		user_bias[i]=0;
	REP(i,M)
		item_bias[i]=0;

	REPM(it,R){
		int i=(*it).F;
		REPM(jt,(*it).S){	
			int j = (*jt).F;
			ratings_mean+= R[i][j];
			if(R[i][j]>10)
				cout<<R[i][j]<<endl;
			ratings_count++;
		}
	}
	ratings_mean= ratings_mean/ratings_count;
	overall_mean = ratings_mean;
	// cout<<"overall_mean: "<<overall_mean<<endl;
	REPM(it,R){
		int i=(*it).F;
		REPM(jt,(*it).S){	
			int j = (*jt).F;
			deviation+= pow((R[i][j] - ratings_mean),2);
		}
	}
	deviation = deviation/ratings_count;
	deviation = sqrt(deviation);
	overall_deviation = deviation;
	// cout<<"overall_deviation: "<<overall_deviation<<endl;
	//Calculate user deviation
	REPM(it,R){
		int i=(*it).F;
		int count=SZ((*it).S);
		double deviation_sum = 0;
		REPM(jt,(*it).S){	
			int j = (*jt).F;
			deviation_sum += (R[i][j]- ratings_mean ) ;
		}
		user_bias[i] = (deviation_sum/double(count));
		// cout<<"bias user "<<i<<": "<<user_bias[i]<<endl;
	}

	//Calculate item bias
	vector<double> sum_items(M,0);	
	vector<double> count_item(M,0);
	REPM(it,R){
		int i=(*it).F;
		REPM(jt,(*it).S){	
			int j = (*jt).F;
			sum_items[j]+= (R[i][j]- ratings_mean );
			count_item[j]++;
		}
	}
	REP(j,M){
		item_bias[j] = (sum_items[j]/count_item[j]);
		// cout<<"bias item "<<j<<": "<<user_bias[j]<<endl;
	}	
}


//--------------------------/
//  FOLKSONOMY-ENHANCED MF
//--------------------------/

void MatrixFactorization::modelLatentVariablesFEMF(){
	cout<<"Starting gradient descent Folksonomy-Enhanced Matrix Factorization"<<endl;
	int totalN = SZ(R);
	int totalM = M;

	calculate_bias();
	content_analyzer();

	// Initializes latent matrixes with random value.
	U_matrix = dlib::randm(totalN,K);
	V_matrix = dlib::randm(K,totalM);	

	REP(i,totalN){
		REP(j,K){
			U_matrix(i,j)=0.0001;
		}
	}
	REP(i,K){
		REP(j,totalM){
			V_matrix(i,j)=0.0001;
		}
	}

	// Move to gradient of function for each prediction made
	double previous_error = 10000000;	
	REP(step,steps){				
		vector<double> tagNormalization(M,0);
		REP(j,M){		
			if(docSimilarities.find(j)!=docSimilarities.end()){
				REP(l,SZ(docSimilarities[j])){
					dlib::matrix<double> Vi,Vl;
					Vi = dlib::colm(V_matrix,j);
					Vl = dlib::colm(V_matrix,docSimilarities[j][l].F);			
					double dif=0;
					REP(latentD,K){
						dif+= abs(Vi(latentD)-Vl(latentD));
					}
					// if(docSimilarities[j][l].S>=similarities_mean){
					tagNormalization[j]+= dif *docSimilarities[j][l].S;						
					// }
				}									
			}
			if(tagNormalization[j]!=0)
				tagNormalization[j] = log2(tagNormalization[j]);	
		}
		REPM(it,R){
			REPM(jt,(*it).S){
				int i = (*it).F; int j = (*jt).F;								
				dlib::matrix<double> A,B;
				A = dlib::rowm(U_matrix,i);
				B = dlib::colm(V_matrix,j);				
				double dotProduct = dlib::dot(A,B);				
				double errorij = R[i][j] - (overall_mean +  user_bias[i] + item_bias[j] + dotProduct);
				
				REP(k,K){				
					double gradientU = (errorij * V_matrix(k,j) - (beta * U_matrix(i,k)));
					double gradientV = (errorij * U_matrix(i,k) - (beta * V_matrix(k,j)) - (beta_tags * tagNormalization[j]) );
					U_matrix(i,k) = U_matrix(i,k) + alpha * (gradientU); 
					V_matrix(k,j) = V_matrix(k,j) + alpha * (gradientV);
					user_bias[i] = user_bias[i] + alpha_bias_user * ( errorij - (beta_bias_user * user_bias[i]));
					item_bias[j] = item_bias[j] + alpha_bias_item * ( errorij - (beta_bias_item * item_bias[j]));
				}
			}			
		}				
		// cout<<"Step : "<<step<<endl;
	}
}

void MatrixFactorization::content_analyzer(){
	// cout<<"Started content_analyzer()"<<endl;
	json itensJSON = json::parse_file(jsonFileName);
	itemContent.resize(M);	

	//Gets TF and Count DF.
	int countExistentTags=0;
	REP(i,itensJSON.size()){
	  try{
	    json& item = itensJSON[i];
	    if(itemStringToID.find(item["isbn"].as<string>()) != itemStringToID.end()){
	    	countExistentTags++;
	   	 	itemContent[itemStringToID[item["isbn"].as<string>()]] = item; 	   	 	
	   	}	   	
	    for (auto it = item["tags"].begin_members(); it != item["tags"].end_members(); ++it){
	    	if(docsWithTerm.find(it->name()) == docsWithTerm.end()){
	    		docsWithTerm[it->name()] = 1;
	    	}else{
	    		docsWithTerm[it->name()] ++;
	    	}
      }
	  }catch(const json_exception& e){}
	}
	cout<<"countExistentTags: "<<countExistentTags<<endl;
	// cout<<docsWithTerm<<endl;

	//Calculate TF-DF
	itemTermWeights.resize(M);
	int TOTAL_DOCS = M;
	REP(i,M){		
		try{
			for (auto it = itemContent[i]["tags"].begin_members(); it != itemContent[i]["tags"].end_members(); ++it){			
				// cout<<stoi(it->value().as<string>())<<endl;
				int freq = stoi(it->value().as<string>());
				double tf = (1.0 + log2(freq));			
				int Ni = docsWithTerm[it->name()];						
				double idf = log2(double(TOTAL_DOCS/Ni));		
				double tf_idf = ( tf * idf);			
				itemTermWeights[i][it->name()] = tf_idf;
			}
		}catch(const json_exception& e){} //cout<<itemContent[i]<<endl;std::cerr << e.what() << std::endl;}
	}	
	// cout<<itemTermWeights<<endl;

	//Calculate similarities.
	// cout<<"Calculating similarities: "<<endl;	
	int countSim=0;

	// RETRIEVING FROM FILE
	// ifstream simFile;
	// simFile.open("similarities.txt");
	// string line;
	// // cout<<"Reading from file similarities.txt."<<endl;
	// int count=0;
	// while(simFile>>line){
	// 	count++;
	// 	int itemFrom = stoi(split(line,'#')[0]);
	// 	string to = split(line,'#')[1];
	// 	vector<string> pairs = split(to,',');
	// 	REP(i,SZ(pairs)){
	// 		int itemTo =  stoi(split(pairs[i],':')[0]);
	// 		double sim =  stod(split(pairs[i],':')[1]);						
	// 		docSimilarities[itemFrom].PB(MP(itemTo,sim));
	// 		similarities_mean+=sim;
	// 		countSim++;
	// 	}
	// }	

	//CALCULATING AND OUTPUTING TO FILE
	float tagsPercentage = 0.5;
	// cout<<"Items sim: "<<(int)(tagsPercentage*M)<<endl;
	// ofstream simFile;
	// simFile.open("similarities.txt");
	REP(i,M){	
		//if(i==(int)(tagsPercentage*M))
		 // break;
		// cout<<"item : "<<i<<endl;		
		// if(SZ(itemTermWeights[i])==0)
			// continue;		

		// simFile<<i<<"#";
		REP(j,M){			
			if(i!=j){
				if(SZ(itemTermWeights[j])==0)				
					continue;						
				double sim = calculateSimilarity(itemTermWeights[i],itemTermWeights[j]);
				if(sim>0.5){
					similarities_mean+=sim;
					countSim++;
					// cout<<countSim<<endl;
					// cout<<itemTermWeights[i]<<endl;
					// cout<<itemTermWeights[j]<<endl;
					// cout<<sim<<endl<<endl<<endl;
					// simFile<<j<<":"<<sim<<",";										
					docSimilarities[i].PB(MP(j,sim));					
				}
			}
		}
		if(countSim>20000)
			break;		
		// simFile<<endl;
	}
	// simFile.close();	


	similarities_mean=similarities_mean/(double)countSim;
	// cout<<"Sim mean: "<<similarities_mean<<endl;
}

double MatrixFactorization::calculateSimilarity(unordered_map<string,double> map1, unordered_map<string,double> map2){
	//From map to arrays:
  vector<double> A,B;   

  int cnt = 0;
	REPM(it,map1){
		string stringID = (*it).F;			
		A.PB((*it).S);
		if(map2.find(stringID) == map2.end())
			B.PB(0);
		else{
			cnt++;
			B.PB(map2[stringID]);
		}
	}
	REPM(it,map2){
		string stringID = (*it).F;			
		B.PB((*it).S);
		if(map1.find(stringID) == map1.end())
			A.PB(0);
		else{
			cnt++;
			A.PB(map1[stringID]);
		}
	}
  //No matching term
  if(cnt == 0)
  	return 0;

	//Calculate similarity according to desired metric
  double dot = 0.0, denom_a = 0.0, denom_b = 0.0 ;

  //Calculate Pearson 
 //  double mean_a = 0.0,mean_b =0.0;
 //  REP(i,SZ(A)){
 //  	mean_a+=A[i];
 //  }
 //  mean_a=mean_a/(double)SZ(A);

 //  REP(i,SZ(B)){
 //  	mean_b+=B[i];
 //  }
 //  mean_b=mean_b/(double)SZ(B);
 // for(unsigned int i = 0u; i < SZ(A); ++i){
 //    dot += (A[i]- mean_a) * (B[i] -mean_b);
 //    denom_a += (A[i]- mean_a)* (A[i]- mean_a) ;
 //    denom_b += (B[i] -mean_b) * (B[i] -mean_b);
	// }  
 //  return dot / (sqrt(denom_a) * sqrt(denom_b));
	// double sigmoidAprox = dot / (sqrt(denom_a) * sqrt(denom_b));
	// sigmoidAprox = sigmoidAprox/(1+ abs(sigmoidAprox));
	// return sigmoidAprox;


	// Calculate cosine similarity
  for(unsigned int i = 0u; i < SZ(A); ++i){
      dot += A[i] * B[i] ;
      denom_a += A[i] * A[i] ;
      denom_b += B[i] * B[i] ;
  }  
  return dot / (sqrt(denom_a) * sqrt(denom_b)) ;

  //Calculate Jaccard
  // double denom = SZ(A);
  // for(unsigned int i = 0u; i < SZ(A); ++i){
  // 	if(A[i]>0 && B[i]>0)
  // 		dot+=1 ;
  // }
  // return dot/denom;

  //Euclidean
  // for(unsigned int i = 0u; i < SZ(A); ++i){
  // 	dot+= ((A[i]-B[i]) * (A[i]-B[i]));
  // }
  // return sqrt(dot);
}

vector<string> MatrixFactorization::split(string str, char delimiter) {
  vector<string> internal;
  stringstream ss(str); // Turn the string into a stream.
  string tok;
  
  while(getline(ss, tok, delimiter)) {
	internal.push_back(tok);
  }
  
  return internal;
}


//--------------------------------------------------------------------------------------//
//    => 	Evaluation functions																													//
//																																											//
//--------------------------------------------------------------------------------------//

/*******************************************************************\
MatrixFactorization::calculateRMSE()

	parameters:
		=> predictions, containing predictions made.
		=> test, part of the data to be calculated error

	return:
		=> RMSE, the error between predictions and the test
/********************************************************************/
double MatrixFactorization::calculateRMSE(vector<double> predictions, vector<double> test){
	double RMSE = 0;
	int n=0;
	REP(i,SZ(test)){
		double error = predictions[i]- test[i];
		RMSE+= pow(error,2.0);
		n++;
	}
	RMSE = RMSE/double(n);
	RMSE = sqrt(RMSE);
	return RMSE;
}
