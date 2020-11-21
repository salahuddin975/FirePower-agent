host_name="ssalahud@rnslab.hpcc.uh.edu:"

path1="/home/ssalahud/fire-power/FirePower-agent-private/database_seed_"
path2="/test_result/"
file_name="fire_power_reward_list_v0.csv"

seed_from=50
seed_to=65
save_dir="rnslab1/"


echo "$path"

for ((seed=seed_from; seed<seed_to; seed++))
do
	path="${host_name}${path1}${seed}${path2}${file_name}"
	scp $path "${save_dir}seed_${seed}_${file_name}"

done


