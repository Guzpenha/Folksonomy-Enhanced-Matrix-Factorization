from IPython import embed
from awesome_print import ap
import json


f = open('../DATASET/ml-latest-small/tags.csv','r')
movies = {}
for line in f:
	isbn = line.split(',')[1] 
	tag = line.split(',')[2]	
	if isbn not in movies.keys():
		movies[isbn]={
			'tags': {
				tag: 1
			}
		}	
	else:
		if(tag not in movies[isbn]['tags'].keys()):
			movies[isbn]['tags'][tag]=1
		else:
			movies[isbn]['tags'][tag] +=1
# print movies
i=0
tags = []
for k in movies.keys():
	tags.append({
		"isbn": k,
		"tags": movies[k]["tags"]
		})

with open("MLcontent.csv", 'wb') as outfile:
    json.dump(tags, outfile)