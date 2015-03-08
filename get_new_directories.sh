

#echo 'Directories newer than ' $1 ' minutes'
find . -maxdepth 1 -type d -cmin -$1 > fns

num=$(cat fns | wc -l)
num2=$((num - 1))
#echo $num2
tail -n $num2 fns
