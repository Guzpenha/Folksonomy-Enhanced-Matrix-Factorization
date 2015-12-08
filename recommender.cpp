/*
  @author: guzpenha@dcc.ufmg.br
*/

#include "MatrixFactorization.h"
#include "Util.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <numeric>
#include <fstream>
#include <ctime>
#include <cstdlib> 
#include "prettyprint.hpp"
using namespace std;

#define PVF(X) for(int x=0;x<X.size();x++){printf("%f ",X[x]);}printf("\n");
#define SZ(X) ((int)(X).size())
#define REP(I, N) for (int I = 0; I < (N); ++I)
#define REPP(I, A, B) for (int I = (A); I < (B); ++I)
#define REPM(I, N) for(auto I = (N.begin()); I != N.end();I++)
#define MP make_pair
#define PB push_back
#define F first
#define S second
#define TAXA_AMOSTRAL 1

// This function returns true if the first pair is "less"
// than the second one according to some metric
// In this case, we say the first pair is "less" if the first element of the first pair
// is less than the first element of the second pair
bool pairCompare(const std::pair<string, int>& firstElem, const std::pair<string, int>& secondElem) {
  return firstElem.S < secondElem.S;
}

int main(int argc, char** argv){	
	//Read file and parse to variables.
	string name_ratings(argv[1]); string name_targets(argv[2]); string name_submission(argv[3]); string jsonName(argv[4]); string line;			
	jsonName = "MLcontent.csv"	;
	double meanFunkSVD,meanFEMF;
	meanFunkSVD=meanFEMF=0.0;
	double bestSVD,bestFEMF;
	bestSVD = bestFEMF = 100;	
	REPP(i,0,5){		
		std::stringstream sstm;
		sstm << "MLratings" << i<<".csv";
		// sstm << "ratings" << i<<".csv";
		string ratings_name = sstm.str();
		cout<<"Fold: "<<ratings_name<<endl;

		ifstream file_ratings;
	  // file_ratings.open("MLratings0.9.csv");
	  file_ratings.open(ratings_name);
	  // file_ratings.open("ratings60.csv");
	  // file_ratings.open("ratings40.csv");
	  file_ratings>>line;//first line

	  vector<string> users;
	  vector<string> itens;
	  vector<int> avaliacoes;  
		while(file_ratings>>line){	
				vector<string> splited = split(line,',');
				users.PB(split(splited[0],':')[0]);
				itens.PB(split(splited[0],':')[1]);	
				avaliacoes.PB(stoi(splited[1]));				
		}

		//Amostragem aleatoria
		vector<int> reading_order;
		REP(i,SZ(avaliacoes))
			reading_order.PB(i);
		srand ( unsigned ( std::time(0) ) );
		// random_shuffle(reading_order.begin(),reading_order.end());
		reading_order.resize(SZ(avaliacoes)*TAXA_AMOSTRAL);

		//From variables to my data structure
		unordered_map<int, unordered_map<int, double> > ratings;
		unordered_map<string,int> uniq_users;
		unordered_map<string,int> uniq_itens;
		int count_users =0;
		int count_itens =0;
		REP(j,SZ(reading_order)){
			int i = reading_order[j];
			if(uniq_users.find(users[i]) == uniq_users.end())	
				uniq_users[users[i]] = count_users++;
			if(uniq_itens.find(itens[i]) == uniq_itens.end())	{
				uniq_itens[itens[i]] = count_itens++;
			}
			ratings[uniq_users[users[i]]][uniq_itens[itens[i]]] = double(avaliacoes[i]);
		}		
		// cout<<uniq_itens<<endl;

		int total_ratings = 0;
		REPM(it,ratings){
			int i = (*it).F;
			REPM(jt,ratings[i]){
				int j = (*jt).F;
				total_ratings++;
			}
		}	

		std::stringstream sstm2;		
		// sstm2<< "targets" << i<<".csv";
		sstm2<< "MLtargets" << i<<".csv";
		string targets_name = sstm2.str();
		// cout<<targets_name<<endl;

		ifstream targets;
		targets.open(targets_name);
		// targets.open("MLtargets0.9.csv");
		// targets.open("targets40.csv");
		// targets.open("targets60.csv");
		ofstream norm1;
		norm1.open(name_submission+"norm1");
		targets>>line;// Read header
		vector<string> target_user, target_item;
		vector<int> target_user_id, target_item_id;
		vector<double> target_rating;
		
		
		while(targets>>line){	
			vector<string> splited = split(line,':');
			target_user.PB(splited[0]);
			target_item.PB(split(splited[1],',')[0]);
	 		target_user_id.PB(uniq_users[splited[0]]);		 	
			target_item_id.PB(uniq_itens[split(splited[1],',')[0]]);
			target_rating.PB(stoi(split(splited[1],',')[1]));
		}

		//Make model for predictions using Gradient Descent MF
		cout<<"Total ratings used: "<<total_ratings<<endl;
		cout<<"Total users: "<<count_users<<endl;
		cout<<"Total itens: "<<count_itens<<endl<<endl;
		
		clock_t begin_time = clock();

		MatrixFactorization * matrixFactorization = new MatrixFactorization(250,5,ratings,count_itens,count_users,jsonName,uniq_itens);
		matrixFactorization->modelLatentVariablesBADlib(false);
		std::cout <<"Tempo FunkSVD: "<< float( clock () - begin_time ) /  CLOCKS_PER_SEC <<endl;
		begin_time = clock();
		MatrixFactorization * FEMF = new MatrixFactorization(250,5,ratings,count_itens,count_users,jsonName,uniq_itens);
		FEMF->modelLatentVariablesFEMF();
		std::cout <<"Tempo FEMF: "<< float( clock () - begin_time ) /  CLOCKS_PER_SEC <<endl<<endl;

		// cout<<"Printing to files"<<endl;


		vector<double> target_predictions = matrixFactorization->predictRatingsBADlib(target_user_id,target_item_id);
		double rmse = matrixFactorization->calculateRMSE(target_predictions,target_rating);

		vector<double> target_predictions_FEMF = FEMF->predictRatingsBADlib(target_user_id,target_item_id);
		double rmse_femf = FEMF->calculateRMSE(target_predictions_FEMF,target_rating);

		cout<<"RMSE MF: "<<rmse<<endl;
		cout<<"RMSE FEMF: "<<rmse_femf<<endl;
		cout<<"================="<<endl<<endl;
		meanFunkSVD+=rmse;
		meanFEMF+=rmse_femf;
		// if(bestSVD>rmse)
			// bestSVD = rmse;
		// if(bestFEMF>rmse_femf)
			// bestFEMF = rmse_femf;


		//nDCG
		unordered_map< string, vector< pair< string,double >  > > listsFEMF, listsFunkSVD;
		unordered_map<string,int> itemRanks;
		REP(i,SZ(target_user)){
			listsFEMF[target_user[i]].PB(MP(target_item[i],target_predictions_FEMF[i]));
			listsFunkSVD[target_user[i]].PB(MP(target_item[i],target_predictions[i]));
			itemRanks[target_item[i]] = target_rating[i];			
		}

		std::stringstream sstm3;		
		sstm3<< "ndcg" << i<<".txt";
		string ndcgName = sstm3.str();

		ofstream ndcg_file;
  	ndcg_file.open(ndcgName);
		REPM(it,listsFEMF){
			sort(listsFEMF[(*it).F].begin(),listsFEMF[(*it).F].end(),pairCompare);
			sort(listsFunkSVD[(*it).F].begin(),listsFunkSVD[(*it).F].end(),pairCompare);
			REP(i,SZ(listsFEMF[(*it).F])){
				ndcg_file << itemRanks[listsFEMF[(*it).F][i].F] <<",";
			}
			ndcg_file<<"#";			
			REP(i,SZ(listsFunkSVD[(*it).F])){
				ndcg_file << itemRanks[listsFunkSVD[(*it).F][i].F] <<",";
			}
			ndcg_file<<endl;

		}

	}
	cout<<endl<<endl<<"Mean FunkSVD: "<<meanFunkSVD/5.0<<endl;
	cout<<"Mean FEMF: "<<meanFEMF/5.0<<endl;
	// cout<<"Best FunkSVD: "<<bestSVD<<endl;
	// cout<<"Best FEMF: "<<bestFEMF<<endl;
	// }
	// vector<double> target_predictions = matrixFactorization->predictRatingsDlib(target_user_id,target_item_id);
	// vector<double> normalized_predictions = matrixFactorization->normalizePrediction(target_predictions,1);
	// ofstream output;
 //  output.open(name_submission);
	// output<<"UserId:ItemId,Prediction\n";
	// norm1<<"UserId:ItemId,Prediction\n";
	// REP(i,SZ(target_user)){		
	// 	output<<target_user[i]<<":"<<target_item[i]<<","<<target_predictions[i]<<endl;
	// 	norm1<<target_user[i]<<":"<<target_item[i]<<","<<normalized_predictions[i]<<endl;
	// }
	// norm1.close();
	// output.close();
	
	return 0;
}




	//CROSS VALIDATION
	// double meanRMSE =0;
	// double meanRMSEN =0;
	// REP(slot,5){
	// unordered_map<int, unordered_map<int, double> > train;
	// unordered_map<int, unordered_map<int, double> > test;
	// MatrixFactorization * matrixFactorization = new MatrixFactorization(250,2,ratings,count_itens,count_users);		
	// matrixFactorization->splitRatings(ratings,train,test,slot);
	// matrixFactorization->modelLatentVariablesBADlib(true);	
	// // matrixFactorization->modelLatentVariablesBA(false);	
	// // return 0;
	
	// //test
	// vector<int> target_user_id, target_item_id;
	// vector<double> actual_values;
	// REPM(it,test){
	// 	int i=(*it).F;
	// 	REPM(jt,(*it).S){	
	// 		int j = (*jt).F;
	// 		target_user_id.PB(i);
	// 		target_item_id.PB(j);
	// 		actual_values.PB(test[i][j]);
	// 	}
	// }
	// // vector<double> target_predictions = matrixFactorization->predictRatingsBADlib(target_user_id,target_item_id);
	// vector<double> target_predictions = matrixFactorization->predictRatingsBA(target_user_id,target_item_id);
	// cout<<"slot: "<<slot<<endl;
	// double rmse = matrixFactorization->calculateRMSE(target_predictions,actual_values);
	// meanRMSE+=rmse;
	// cout<<"RMSE: "<<rmse<<endl;
	// vector<double> normalized_predictions = matrixFactorization->normalizePrediction(target_predictions,1);
	// rmse = matrixFactorization->calculateRMSE(normalized_predictions,actual_values);
	// meanRMSEN+=rmse;
	// cout<<"RMSE norm trim: "<<rmse<<endl;
	// vector<double> normalized_predictions2 = matrixFactorization->normalizePrediction(target_predictions,2);
	// rmse = matrixFactorization->calculateRMSE(normalized_predictions2,actual_values);
	// cout<<"RMSE norm range: "<<rmse<<endl;
	// vector<double> normalized_predictions3 = matrixFactorization->normalizePrediction(target_predictions,3);
	// rmse = matrixFactorization->calculateRMSE(normalized_predictions3,actual_values);
	// cout<<"RMSE norm zScore: "<<rmse<<endl;


	// cout<<"-----------\n\n"<<endl;
	// }
	// cout<<"====================="<<endl<<endl;
	// cout<<"MEAN RMSE: "<<meanRMSE/5.0<<endl;
	// cout<<"MEAN RMSE normalized: "<<meanRMSEN/5.0<<endl;
	// return 0;
// END CROSS VALIDATION

	//Steps 
	// REPP(i,151,2500){
	// 	train.clear();
	// 	test.clear();
	// 	actual_values.clear();
	// 	target_predictions.clear();
	// 	normalized_predictions.clear();
	// 	rmse=0;
	// 	matrixFactorization= new MatrixFactorization(i,10,ratings,count_itens,count_users);	
	// 	matrixFactorization->splitRatings(ratings,train,test,0);			
	// 	REPM(it,test){
	// 		int i=(*it).F;
	// 		REPM(jt,(*it).S){	
	// 			int j = (*jt).F;
	// 			target_user_id.PB(i);
	// 			target_item_id.PB(j);
	// 			actual_values.PB(test[i][j]);
	// 		}
	// 	}
	// 	matrixFactorization->modelLatentVariablesBADlib(true);
	// 	target_predictions = matrixFactorization->predictRatingsBADlib(target_user_id,target_item_id);
	// 	normalized_predictions = matrixFactorization->normalizePrediction(target_predictions,1);
	// 	rmse = matrixFactorization->calculateRMSE(normalized_predictions,actual_values);
	// 	cout<<i<<","<<rmse<<endl;		
	// }
	// return 0;


	// MFLIB START

	// matrixFactorization= new MatrixFactorization(5000,100,ratings,count_itens,count_users);	
	// matrixFactorization->splitRatings(ratings,train,test,0);			
	// REPM(it,test){
	// 	int i=(*it).F;
	// 	REPM(jt,(*it).S){	
	// 		int j = (*jt).F;
	// 		target_user_id.PB(i);
	// 		target_item_id.PB(j);
	// 		actual_values.PB(test[i][j]);
	// 	}
	// }
	// matrixFactorization->modelLatentVariablesDlib(true);
	// target_predictions = matrixFactorization->predictRatingsBADlib(target_user_id,target_item_id);
	// normalized_predictions = matrixFactorization->normalizePrediction(target_predictions,1);
	// rmse = matrixFactorization->calculateRMSE(normalized_predictions,actual_values);
	// cout<<"100,"<<rmse<<endl;
	// return 0;
	// matrixFactorization= new MatrixFactorization(5000,200,ratings,count_itens,count_users);	
	// matrixFactorization->splitRatings(ratings,train,test,0);			
	// REPM(it,test){
	// 	int i=(*it).F;
	// 	REPM(jt,(*it).S){	
	// 		int j = (*jt).F;
	// 		target_user_id.PB(i);
	// 		target_item_id.PB(j);
	// 		actual_values.PB(test[i][j]);
	// 	}
	// }
	// matrixFactorization->modelLatentVariablesDlib(true);
	// target_predictions = matrixFactorization->predictRatingsBADlib(target_user_id,target_item_id);
	// normalized_predictions = matrixFactorization->normalizePrediction(target_predictions,1);
	// rmse = matrixFactorization->calculateRMSE(normalized_predictions,actual_values);
	// cout<<"200,"<<rmse<<endl;
	// return 0;
	// // MFLIB END
