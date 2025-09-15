#!/bin/bash
#SBATCH -J QuantumIA                        # Nome do job
#SBATCH -o %a.out                           # Nome do arquivo de saída (%j = ID do JOB, %a = ID da Tarefa no Array)
#SBATCH -t 11:00:00                            # Tempo de execução (hh:mm:ss) - 15 minutos
#SBATCH --mail-user=gubio@estudante.ufscar.br
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --partition=fast
#SBATCH --array=1-101                           # Definindo um array de jobs de 1 a 10

# Definição das pastas de input/output e job local para cada tarefa
container_in=/opt/input                         # Pasta no cluster para input
container_out=/opt/output                       # Pasta no cluster para output
local_job="/scratch/job.${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"   # Pasta temporária local para cada tarefa
local_in="${local_job}/input/"                  # Pasta local para arquivos de entrada
local_out="${local_job}/output/"                # Pasta local para arquivos de saída
remote_drive="cloud:/"         			 # Caminho no Google Drive configurado no rclone

mysrun="srun "

# Função para limpar o ambiente ao finalizar o job
function clean_job() {
  echo "Limpando ambiente da tarefa ${SLURM_ARRAY_TASK_ID}..."
  ${mysrun} rm -rf "${local_job}"
}
trap clean_job EXIT HUP INT TERM ERR
set -eE
umask 077

# Início do job
echo "Executando tarefa ${SLURM_ARRAY_TASK_ID}..."

${mysrun} mkdir -p "${local_in}"
${mysrun} mkdir -p "${local_out}"


echo "Executando no container..."
srun --mpi=pmi2 singularity run \
     --bind=/scratch:/scratch \
     --bind=/var/spool/slurm:/var/spool/slurm \
     Singularity.simg \
     python3 /test.py ${SLURM_ARRAY_TASK_ID} # Executa com ID da tarefa como argumento


echo "Finalizado ${SLURM_ARRAY_TASK_ID} "


