#include "discretization.h"
#include "level_structures.h"//for all the objects that are needed for a run.
#include "time_loop.h"
#include <deal.II/base/mpi.h>
#include "euler/description.h"
#include "euler/hyperbolic_system.h"
#include "my_app.h"
#include "state_vector.h"
#include "mgrit_functions.template.h"



/**
 * Right now, this executable runs a simulation equivalent to a ryujin run.
*/
using StateVector = mgrit::MyApp<double, ryujin::Euler::Description, 2>::StateVector;
using App = mgrit::MyApp<double, ryujin::Euler::Description, 2>;

constexpr int dim = 2;

int main(int argc, char *argv[]){

  const std::string prm_name = argv[1];//Decide on how to make parameter set in test format.
  const std::vector<int> refinement_levels = {1, 3, 5};

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);  //create objects
  const MPI_Comm comm_world = MPI_COMM_WORLD;
  
  mgrit::MyApp<double, ryujin::Euler::Description, 2> app(comm_world, comm_world, refinement_levels);

  app.initialize(prm_name);

  // Set up data.
  mgrit::MyVector<double, ryujin::Euler::Description, 2> fineU, fineCopy,
                                                         middleU,
                                                         coarseU, coarseCopy;

  // Initialize data needs to be at t = 0 on the fine level.
  ryujin::Vectors::reinit_state_vector<ryujin::Euler::Description>(fineU.U,
								   *(app.levels[0]->offline_data));
  std::get<0>(fineU.U) = app.levels[0]->initial_values->get().interpolate_hyperbolic_vector(0.0);

  // now, we need to initialize each vector to the appropriate level.
  app.reinit_to_level(&middleU, 1);
  app.reinit_to_level(&coarseU, 2);
  app.reinit_to_level(&coarseCopy, 2);

  // now interpolate the fine to the coarse.
  app.interpolate_between_levels(coarseU, 2, fineU, 0);
  // and the fine to the middle, then the middle to the coarse copy.
  app.interpolate_between_levels(middleU, 1, fineU, 0);
  app.interpolate_between_levels(coarseCopy, 2, middleU, 1);

  //compare that the single fine->coarse = fine->middle->coarse
  std::get<0>(coarseU.U) -= std::get<0>(coarseCopy.U);
  double diff = std::get<0>(coarseU.U).l2_norm();

  if (diff < 1e-10)
  {
    std::cout << "Interpolation fine->coarse consistent." << std::endl;
  }
  else
  {
    std::cout << "Fine->Coarse Not OK." << std::endl;
  }

  // Now compare interpolation coarse->fine
  app.reinit_to_level(&fineCopy, 0);
  app.reinit_to_level(&middleU, 1);
  app.reinit_to_level(&coarseU, 2);
  // Initialize data needs to be at t = 0 on the fine level. We test by 
  ryujin::Vectors::reinit_state_vector<ryujin::Euler::Description>(coarseU.U,
								   *(app.levels[2]->offline_data));
  std::get<0>(coarseU.U) = app.levels[2]->initial_values->get().interpolate_hyperbolic_vector(0.0);

  // now interpolate the coarse to fine.
  app.interpolate_between_levels(fineU, 0, coarseU, 2);
  // and then coarse to the middle, then the middle to the fine copy.
  app.interpolate_between_levels(middleU, 1, coarseU, 2);
  app.interpolate_between_levels(fineCopy, 0, middleU, 1);

  //compare that the single coarse->fine = coarse->middle->fine
  std::get<0>(fineU.U) -= std::get<0>(fineCopy.U);
  diff = std::get<0>(fineU.U).l2_norm();

  if (diff < 1e-10)
  {
    std::cout << "Interpolation coarse->fine consistent." << std::endl;
  }
  else
  {
    std::cout << "Coarse->Fine Not OK." << std::endl;
  }

  return 0;
}
