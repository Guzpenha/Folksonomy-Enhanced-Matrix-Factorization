require 'pry'
file = File.new("/home/Guz/Dropbox/UFMG/7º Período/POC/DATASET/ml-latest-small/ratings.csv", "r")

train_size = 0.9
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

# normal
puts ratings.size
train = ratings[0..(train_size*ratings.size)]
test =  ratings[(train_size*ratings.size)+1..-1]
# binding.pry

file_train = File.new("MLratings#{train_size}.csv",'w')
file_train.write("Header\n")
train.each do |line|
  file_train.write(line)
end
file_train.close()


file_test = File.new("MLtargets#{train_size}.csv",'w')
file_test.write("Header\n")
test.each do |line|   
  file_test.write(line)
end
file_test.close()

# 5-Fold
# (0..4).each do |i|
#   # puts ratings.size
#   puts "Fold  #{i}"
#   test = ratings[(i*test_size*count)..((i+1)*test_size*count)]
#   del_ratings = ratings.clone 
#   puts "from: #{(i*test_size*count).to_i}"
#   puts "to : #{ ((i+1)*test_size*count).to_i}"
#   if(i==4) 
#     train = ratings[0..(i*test_size*count).to_i]
#   else
#     ((i*test_size*count).to_i..((i+1)*test_size*count).to_i).map { |j| del_ratings.delete_at j }
#     train =  del_ratings
#   end
#   puts "train size: #{train.size}"

#   file_train = File.new("MLratings#{i}.csv",'w')
#   file_train.write("Header\n")
#   train.each do |line|
#   	file_train.write(line)
#   end
#   file_train.close()


#   file_test = File.new("MLtargets#{i}.csv",'w')
#   file_test.write("Header\n")
#   test.each do |line|		
#   	file_test.write(line)
#   end
#   file_test.close()
# end