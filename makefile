all: TP1

TP1: 
	g++ -o recommender recommender.cpp MatrixFactorization.cpp -std=c++11 -O3 -Iinclude 

run:
	./recommender ratings.csv targets.csv submission.csv content.csv
clean:
	rm -rf *.o
