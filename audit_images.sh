#!/bin/bash
read -p "Please enter path:" path

for i in $path/*.jpg;
do xdg-open $i && read -p "Bad?" ANS

if [ ${i:0:4} == 'good' ]
then
    continue

elif [ $ANS == 'y' ]
then
    jpg_name=${i//$path\//}
    mv "$path/$jpg_name" "$path/bad_$jpg_name"
    mv "$path/bad_$jpg_name" $path/bad_imgs
    pkill xviewer

elif [ $ANS == 'n' ]
then
    jpg_name=${i//$path\//}
    mv "$path/$jpg_name" "$path/good_$jpg_name"
    pkill xviewer

elif [ $ANS == 'quit' ]
then
    exit 0
fi;

done
