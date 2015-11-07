require 'pry'
file = File.new("/home/Guz/Dropbox/UFMG/7º Período/POC/DATASET/BX-CSV-Dump/BX-Book-Ratings.csv", "r")

train_size = 0.8
test_size = 1-train_size
count =0

ratings = []
file.gets
while (line = file.gets)
    if ! line.valid_encoding?
  		line = line.encode("UTF-16be", :invalid=>:replace, :replace=>"?").encode('UTF-8')
		end
    user,isbn,rating = line.gsub("\"",'').strip.split(";")
    if rating !="0"
    	# puts "#{user}, #{isbn}, #{rating}"
   		ratings << "#{user}:#{isbn},#{rating}\n"
    	count+=1
    end
end
file.close

ratings = ratings.shuffle
train =  ratings[0..(train_size*ratings.size())]
test = ratings[(train_size*ratings.size())..-1]
puts count

file_train = File.new("ratings.csv",'w')
file_train.write("Header\n")
train.each do |line|
	file_train.write(line)
end
file_train.close()


file_test = File.new("targets.csv",'w')
file_test.write("Header\n")
test.each do |line|		
	file_test.write(line)
end
file_test.close()

# binding.pry