#include "discretization.h"
#include "euler/description.h"
#include "euler/hyperbolic_system.h"
#include "level_structures.h" //for all the objects that are needed for a run.
#include "my_app.h"
#include "state_vector.h"
#include "time_loop.h"
#include <deal.II/base/mpi.h>

// Postprocessing
#include "mgrit_functions.template.h"

bool is_print_time(mgrit::MyApp<NUMBER, ryujin::Euler::Description, 2> &app,
                   const NUMBER time)
{
  // get print frequency from app
  const NUMBER pdt = app.c_points()[1];

  // check if this time is close to any of the print times.
  for (int i = 0; i < app.num_bricks + 1; i++) {
    if (std::abs(i * pdt - time) < 1e-11)
      return true;
  }

  return false;
};

/**
 * Right now, this executable runs a simulation equivalent to a ryujin run.
 */
int main(int argc, char *argv[])
{

  const std::string prm_name = argv[1];
  const std::string restart_fname = argv[2];
  const double tstart = std::stof(argv[3]);
  const double tstop = std::stof(argv[4]);
  const unsigned int refinement = std::stoul(argv[5]);

  std::cout << "Restarting computation with file " << restart_fname
            << "\nending at time t in [" << tstart << ", " << tstop << "]."
            << std::endl;

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, 1); // create objects
  mgrit::MyApp<NUMBER, ryujin::Euler::Description, 2> app(
      MPI_COMM_WORLD, MPI_COMM_WORLD, {(int)refinement});
  std::cout << "Initializing with prm = " + prm_name << std::endl;

  app.initialize(prm_name);

  using StateVector =
      mgrit::MyApp<NUMBER, ryujin::Euler::Description, 2>::StateVector;
  // Set up data.
  mgrit::MyVector<NUMBER, ryujin::Euler::Description, 2> U_data;

  /**
   * Define the postprocess lambdas
   */
  static int cycle = 0;
  const auto postprocess = [&]([[maybe_unused]] const StateVector U,
                               double time) {
    Assert(
        &U == &(U_data.U),
        dealii::ExcMessage("The data and the data being stepped need to be the "
                           "same for meaningful postprocessing."));
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "Postprocessing at t=" << time << std::endl;

    // we only want to do postprocessing at C-points
    // then we want to know the forces if we are at a print time.
    if (is_print_time(app, time)) {

      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Projecting and printing: " << time << std::endl;
      // as a first step of postprocessing, we want to imitate the MGRIT
      // algorithm and use a projection.

      mgrit_functions::
          enforce_physicality_bounds<ryujin::Euler::Description, 2, NUMBER>(
              U_data, app.finest_level, app, time);
      const int t_idx = static_cast<int>(time / tstop * app.num_bricks);
      dealii::Tensor<1, 2> forces = mgrit_functions::
          calculate_forces_on_object<NUMBER, ryujin::Euler::Description, 2>(
              &app, U_data, time, cycle, t_idx);
      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Forces[0]=" << forces[0] << " at t=" << time << std::endl;

      app.print_solution(U_data.U, time, 0, restart_fname, cycle++);
    }
  };

  // Initialize data needs to be at t = 0.
  Assert(std::fabs(tstart) < 1.e-6,
         dealii::StandardExceptions::ExcMessage(
             "tstart needs to be zero for this executble."
             "Here, tstart=" +
             std::to_string(tstart)));
  // calls update_ghost_values() and reinits U and precomputed from the
  // state_vector.
  ryujin::Vectors::reinit_state_vector<ryujin::Euler::Description>(
      U_data.U, *(app.levels[0]->offline_data));
  std::get<0>(U_data.U) =
      app.levels[0]->initial_values->get().interpolate_hyperbolic_vector(0.0);

  app.time_loops[0]->change_base_name(restart_fname);
  app.time_loops[0]->set_timer_granularity(app.c_points()[1]);
  // postprocess t=0
  postprocess(U_data.U, tstart);
  // now that we have the data, we call the run function
  app.time_loops[0]->run_with_initial_data(
      U_data.U,
      tstop,
      tstart,
      /*mgrit_specified_printing*/ true,
      postprocess,
      /*print_every_step*/ false,
      /*enforce_granularity_in_substeps*/ true);

  return 0;
}
