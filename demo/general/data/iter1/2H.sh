#PBS -N chenmw
#PBS -l nodes=1:ppn=24
#PBS -l walltime=1500:00:00
#PBS -q all
#PBS -V
#PBS -S /bin/bash 
          
          
source /opt/intel/composer_xe_2015.1.133/bin/iccvars.sh	intel64
source /opt/intel/composer_xe_2015.1.133/bin/ifortvars.sh intel64
source /opt/intel/composer_xe_2015.1.133/mkl/bin/intel64/mklvars_intel64.sh
source /opt/intel/impi/5.0.2.044/intel64/bin/mpivars.sh

cd $PBS_O_WORKDIR
ulimit -s unlimited
EXEC=/opt/vasp5.3.5/vasp

NP=`cat $PBS_NODEFILE | wc -l`
NN=`cat $PBS_NODEFILE | sort | uniq | tee /tmp/nodes.$$ | wc -l`
    
cat $PBS_NODEFILE > /tmp/nodefile.$$

mpdboot -f /tmp/nodefile.$$ -n $NN
#mpirun -genv I_MPI_DEVICE rdma -machinefile /tmp/nodefile.$$ -n $NP $EXEC

mmaps -2d  -c=3 -m=5 -l=/data/mwchen-ICME/latin/ZrTe2_2H_lat.in &
pollmach runstruct_vasp -w /data/mwchen-ICME/wraps/MoTe2_vasp_4000_no_spin.wrap mpirun -machinefile /tmp/nodefile.$$ -np $NP
#cat $PBS_NODEFILE >> test.txt
mpdallexit

rm -f /tmp/nodefile.$$

