#!/bin/bash

# the .prm is $1
# the nWorld is $2
# the nX is $3
# the number of threads is $4
# the rest are the refinement levels.
## warning, these need to be sorted into ascending order eg. 1 3 6 levels of refinement
## since the logic of this file assumes that $5 is the coarsest refinement.

#TODO: make some kind of assertion here?

prm="$1"
nworld="$2"
nx="$3"
nthreads="$4"
LOGNAME="$5"
s_refinements="${@:6:($#)}"
refinements=(${s_refinements})

echo "PRM:${prm}"
echo "NWORLD:${nworld}"
echo "NX:${nx}"
echo "NTHREADS:${nthreads}"
echo "refinements:"${s_refinements}

if [ ${refinements[0]} -lt 4 ]; then
    echo "Using 1 process for startup since the # refinements on coarsest level is small."
    DEAL_II_NUM_THREADS="${nthreads}" mpirun -n 1 startup "${prm}" ${s_refinements}
else
    DEAL_II_NUM_THREADS="${nthreads}" mpirun -n "${nworld}" startup "${prm}" ${s_refinements}
fi
# once the setupfiles are written, we start the regular program.

time DEAL_II_NUM_THREADS="${nthreads}" mpirun -n "${nworld}" cpp_myapp_testing "${prm}" "${nx}" ${s_refinements} | tee "${LOGNAME}"
