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


#scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/FirePower-agent-private/fire_propagation_0_\*.png  .
#scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/FirePower-agent-private/database_seed_10004/test_result/episodic_test_result_ep_400.csv  .
#scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/FirePower-agent-private/gams_feasible.csv   .
#scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/gym-firepower/gym_firepower/envs/gams/temp/\*.zip   .

scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/FirePower-agent-private/database_seed_4/test_result/episodic_test_result_ep_1960.csv  seed_4/
scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/FirePower-agent-private/database_seed_5/test_result/episodic_test_result_ep_1920.csv  seed_5/
scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/FirePower-agent-private/database_seed_6/test_result/episodic_test_result_ep_1920.csv  seed_6/
scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/FirePower-agent-private/database_seed_7/test_result/episodic_test_result_ep_1600.csv  seed_7/
scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/FirePower-agent-private/database_seed_8/test_result/episodic_test_result_ep_1460.csv  seed_8/
scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/FirePower-agent-private/database_seed_9/test_result/episodic_test_result_ep_1620.csv  seed_9/
scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/FirePower-agent-private/database_seed_10/test_result/episodic_test_result_ep_1400.csv  seed_10/
scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/FirePower-agent-private/database_seed_11/test_result/episodic_test_result_ep_1400.csv  seed_11/
scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/FirePower-agent-private/database_seed_12/test_result/episodic_test_result_ep_1400.csv  seed_12/