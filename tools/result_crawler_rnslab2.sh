host_name="ssalahud@rnslab2.hpcc.uh.edu:"

path1="/home/ssalahud/fire_power/FirePower-agent-private/database_seed_"
path2="/test_result/"
file_name="fire_power_reward_list_v0.csv"

seed_from=101
seed_to=121
save_dir="rnslab2/"


echo "$path"

for ((seed=seed_from; seed<seed_to; seed++))
do
	path="${host_name}${path1}${seed}${path2}${file_name}"
	scp $path "${save_dir}seed_${seed}_${file_name}"

done


