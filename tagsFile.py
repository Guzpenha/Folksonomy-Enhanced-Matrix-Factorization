from pymongo import MongoClient
from IPython import embed
from awesome_print import ap
import json

client = MongoClient('mongodb://localhost:27017/')
books_db = client.bookcrossing.books

# embed()

i=0
tags = []
for book in books_db.find({"status_tags": "ISBN_match", "tags": {"$ne": "not_found"} }):
	filtered_tags = {}
	for k in book["tags"].keys():
		if k != "unread" and k!= "to-read" and k!= "read" and k!= "own" and k!="favorite" and k!="paperback" and k!="partially read" and k!= "ebook" and k!="sell" and k!= "goodreads" and k!="donated" and k!= "read in" and k!="owned" and k!="hardcover" and k!="signed" and k!="TBR" and k!="sold" :
			filtered_tags[k] = book["tags"][k]
	tags.append({
		"isbn": book["ISBN"],
		"tags": filtered_tags
		})

with open("content.csv", 'wb') as outfile:
    json.dump(tags, outfile)