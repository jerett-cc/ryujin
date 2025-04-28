#include "discretization.h"
#include "level_structures.h"//for all the objects that are needed for a run.
#include "time_loop.h"
#include <deal.II/base/mpi.h>
#include "euler/description.h"
#include "euler/hyperbolic_system.h"
#include "my_app.h"
#include "state_vector.h"
#include "mgrit_functions.template.h"

#include "braid.hpp"//For braidstatus struct.
#include "braid.h"//For braid_BufferStatus


/**
 * Test if the buffer packing functions in myapp work properly.
 * Create a vector, pack it into a buffer, then unpack it into another vector,
 * then compare the vectors. They should be the same.
 */
using StateVector = mgrit::MyApp<double, ryujin::Euler::Description, 2>::StateVector;
using App = mgrit::MyApp<double, ryujin::Euler::Description, 2>;

constexpr int dim = 2;

int main(int argc, char *argv[]){

  const std::string prm_name = argv[1];//Decide on how to make parameter set in test format.
  const std::vector<int> refinement_levels = {0};

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);  //create objects
  const MPI_Comm comm_world = MPI_COMM_WORLD;

  // Overarching structure.
  mgrit::MyApp<double, ryujin::Euler::Description, 2> app(comm_world, comm_world, refinement_levels);
  app.initialize(prm_name);

  // Set up data in the one to pack.
  mgrit::MyVector<double, ryujin::Euler::Description, 2> packU, unpackU;
  // Initialize data needs to be at t = 0 on the fine level.
  ryujin::Vectors::reinit_state_vector<ryujin::Euler::Description>(packU.U,
								   *(app.levels[0]->offline_data));
  std::get<0>(packU.U) = app.levels[0]->initial_values->get().interpolate_hyperbolic_vector(0.0);

  void* buffer;// Seems wrong.

  //Set up necessar bufferstatus from XBraid.
  braid_BufferStatus b_;
  _braid_BufferStatusInit(0,0,0,0,b_);
  BraidBufferStatus bstatus(b_);

  // app.BufPack((_braid_Vector_struct*)packU, buffer, bstatus);
  // app.BufUnpack(buffer, (_braid_Vector_struct*)unpackU, bstatus);
  
  
  std::get<0>(packU.U) -= std::get<0>(unpackU.U);
  double diff = std::get<0>(packU.U).l2_norm();

  if (diff < 1e-10)
  {
    std::cout << "Pack->Unpack is consistent." << std::endl;
  }
  else
  {
    std::cout << "Pack->Unpack Not OK." << std::endl;
  }

  return 0;
}
