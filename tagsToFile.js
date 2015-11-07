myCursor = db.books.find({status_tags: "ISBN_match", tags: {$ne: "not_found"} })
print(myCursor.size());
var content = [];
while (myCursor.hasNext()){
		// printjson(myCursor.next().ISBN);
		isbn = myCursor.next().ISBN;
		tags = myCursor.next().tags;

		for (var prop in tags) {
        if(prop.indexOf('"') != -1){	
		        // print("Key:" + prop);
		        // tags[prop.replace("\"",'')] = tags[prop];							        	
		        delete tags[prop];		     
		        tags = this.tags; 		       
      			// printjson(tags);  
        }
    }

		content.push({
			isbn: isbn,
			tags: tags
		});
}
printjson(content); 
// print(content.length());
