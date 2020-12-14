seed=101
episode=400

#rnslab1
scp ssalahud@rnslab.hpcc.uh.edu:/home/ssalahud/fire-power/FirePower-agent-private/database_seed_${seed}/trained_model/*_${episode}.h5  ./../database_seed_50/trained_model/
scp ssalahud@rnslab.hpcc.uh.edu:/home/ssalahud/fire-power/FirePower-agent-private/database_seed_${seed}/parameters*.txt  ./../database_seed_50/

#rnslab2
#scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/FirePower-agent-private/database_seed_${seed}/trained_model/*_${episode}.h5  ./../database_seed_50/trained_model/
#scp ssalahud@rnslab2.hpcc.uh.edu:/home/ssalahud/fire_power/FirePower-agent-private/database_seed_${seed}/parameters*.txt  ./../database_seed_50/

