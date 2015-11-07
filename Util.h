/*
  @author: guzpenha@dcc.ufmg.br
*/
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
  
using namespace std;

// split(str, delimiter)
//
// 		args: <str> (string to be splitted) <delimiter> (delimiter to split by)
//
// 
// 		return: (vector of string containing each string splitted by delimiter)
//

vector<string> split(string str, char delimiter) {
  vector<string> internal;
  stringstream ss(str); // Turn the string into a stream.
  string tok;
  
  while(getline(ss, tok, delimiter)) {
	internal.push_back(tok);
  }
  
  return internal;
}