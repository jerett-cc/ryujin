#!/ bin / bash

#the.prm is $1
#the nWorld is $2
#the nX is $3
#the number of threads is $4
#the logname is $5
#the refinement is $6
#the rest are the time integration schemes to use on the levels.
##warning, these will be sorted into ascending order
##eg.if user inputs "erk 11" "erk 54" "erk 33"
##the program will reorder these so that the coarsest level
##is the least computationally costly
##0 : "erk 54", 1 : "erk 33", 2 : "erk 11"

prm="$1"
nworld="$2"
nx="$3"
nthreads="$4"
LOGNAME="$5"
refinement="$6"

echo "PRM:${prm}"
echo "NWORLD:${nworld}"
echo "NX:${nx}"
echo "NTHREADS:${nthreads}"
echo "n_refinements:${refinement}"
echo "integrators:${@:7}"

if [ ${refinement} -lt 6 ]; then
    echo "Using 5 process for startup since the # refinements is small."
    DEAL_II_NUM_THREADS="${nthreads}" mpirun -n 5 startup_same_mesh "${prm}" "${refinement}" "${@:7}"
else
    DEAL_II_NUM_THREADS="${nthreads}" mpirun -n "${nworld}" startup_same_mesh "${prm}" "${refinement}" "${@:7}"
fi

#once the setupfiles are written, we start the regular program.
time DEAL_II_NUM_THREADS="${nthreads}" mpirun -n "${nworld}" mgrit_same_mesh "${prm}" "${nx}" ${refinement} "${@:7}" | tee "${LOGNAME}"

#now that we are done, concatenate the PRM to the LOG.
echo "____________________PRM____________________" >> ${LOGNAME}
cat ${prm} >> ${LOGNAME}
