models=("deepseek/deepseek-reasoner")
datasets=("syn-3bfo" "gb1" "trpb" "aav" "gfp")
tasks=("single")

# Example: get a length if you need it
# length=${#datasets[@]}

for m in "${models[@]}"; do
  for d in "${datasets[@]}"; do
    for t in "${tasks[@]}"; do
      echo "Running: python3 cli.py $t --oracle $d --generations 8 --population-size 200 --offspring-size 100 --model $m"
      nohup python3 cli.py "$t" --oracle "$d" --generations 8 --population-size 200 --offspring-size 100 --model "$m" > "$d"_"$t".log 2>&1 &
    done
  done
done