seed=131
episode=400

#rnslab1
#scp ssalahud@rnslab.hpcc.uh.edu:/home/ssalahud/fire-power/FirePower-agent-private/database_seed_${seed}/trained_model/*_${episode}.h5  ./../database_seed_50/trained_model/
#scp ssalahud@rnslab.hpcc.uh.edu:/home/ssalahud/fire-power/FirePower-agent-private/database_seed_${seed}/replay_buffer/*  ./../database_seed_50/replay_buffer/
#scp ssalahud@rnslab.hpcc.uh.edu:/home/ssalahud/fire-power/FirePower-agent-private/database_seed_${seed}/parameters*.txt  ./../database_seed_50/
#scp ssalahud@rnslab.hpcc.uh.edu:/home/ssalahud/fire-power/FirePower-agent-private/fire-power-${seed}.log  ./../logs/


#rnslab2
#scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/FirePower-agent-private/database_seed_${seed}/trained_model/*_${episode}.h5  ./../database_seed_50/trained_model/
#scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/FirePower-agent-private/database_seed_${seed}/replay_buffer/*  ./../database_seed_50/replay_buffer/
#scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/FirePower-agent-private/database_seed_${seed}/parameters*.txt  ./../database_seed_50/
#scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/FirePower-agent-private/fire-power-${seed}.log  ./../logs/


#scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/FirePower-agent-private/removed_load_database_seed_10032/test_result/\*  /Users/smsalahuddinkadir/Documents/Projects/FirePower/test_results/removed_bus_load/fire_spread_020/seed_10032/
scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/FirePower-agent-private/removed_load_sp25_database_seed_10032/test_result/\*  /Users/smsalahuddinkadir/Documents/Projects/FirePower/test_results/removed_bus_load/fire_spread_025/seed_10032/


#scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/FirePower-agent-private/database_seed_10004/test_result/episodic_test_result_ep_400.csv  .
#scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/FirePower-agent-private/gams_feasible.csv   .
#scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/gym-firepower/gym_firepower/envs/gams/temp/\*.zip   .
