#include "discretization.h"
#include "euler/description.h"
#include "euler/hyperbolic_system.h"
#include "level_structures.h" //for all the objects that are needed for a run.
#include "my_app.h"
#include "state_vector.h"
#include "time_loop.h"
#include <deal.II/base/mpi.h>


/**
 * Right now, this executable runs a simulation equivalent to a ryujin run.
 */
int main(int argc, char *argv[])
{

  const std::string prm_name = argv[1];
  const int nx = std::stoi(argv[2]);
  std::vector<int> refinement_levels(argc - 3);

  for (int i = 3; i < argc; i++)
    refinement_levels[i - 3] = std::stoi(argv[i]);

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, 1); // create objects

  MPI_Comm comm_x, comm_t;
  MPI_Comm world = MPI_COMM_WORLD;
  braid_SplitCommworld(&world,
                       nx /*the number of spatial processors per time brick*/,
                       &comm_x,
                       &comm_t);

  mgrit::MyApp<NUMBER, ryujin::Euler::Description, 2> app(
      comm_x, comm_t, refinement_levels);

  app.initialize(prm_name);

  using StateVector =
      mgrit::MyApp<NUMBER, ryujin::Euler::Description, 2>::StateVector;

  braid_Vector U, V;
  braid_Real normU, normV;
  bool separate = false;

  if (dealii::Utilities::MPI::n_mpi_processes(world) > 1)
    separate = true;

  if (separate && dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
    app.Init(0.0, &U);
    app.SpatialNorm(U, &normU);
    std::cout << "Norm U: " << normU << std::endl;
  } else if (separate &&
             dealii::Utilities::MPI::this_mpi_process(comm_t) == 1) {
    app.Init(0.25, &V);
    app.SpatialNorm(V, &normV);
    std::cout << "normV: " << normV << std::endl;
  } else {

    app.Init(0.0, &U);
    app.Init(0.25, &V);

    app.SpatialNorm(U, &normU);
    app.SpatialNorm(V, &normV);
    std::cout << "Norm U: " << normU << ", normV: " << normV << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  return 0;
}
