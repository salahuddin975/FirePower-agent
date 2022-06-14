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
scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/FirePower-agent-private/gams_feasible.csv   .
scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/remote_compiler/gym-firepower/gym_firepower/envs/gams/temp/\*.zip   .
