make 
cd a3-input
for file in *; do 
    echo $file
    # get file name at the end of $file

    # name = ${file//"a3-input"}
    # echo $name
    ../a3 < $file > ../$file.out
    diff ../$file.out ../a3-output/$file --brief
done
cd ..
rm -rf *.out