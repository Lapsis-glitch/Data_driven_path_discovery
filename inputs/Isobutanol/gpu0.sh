for i in $(seq 0 32)
do
 if [ $i -lt 17 ]; then
  echo $i
  python ./MACE_core.py -s $i -e $(($i+1)) -d cuda:0 > MACE.log &
 fi
 if [ $i -gt 16 ]; then
  echo $i
  python ./MACE_core.py -s $i -e $(($i+1)) -d cuda:1 > MACE.log &
 fi
done
