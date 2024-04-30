#include "discretization.h"
#include "level_structures.h"//for all the objects that are needed for a run.
#include "time_loop.h"
#include <deal.II/base/mpi.h>
#include "euler/description.h"
#include "euler/hyperbolic_system.h"
#include "braid_funcs.h"



/**
 * Right now, this executable runs a simulation equivalent to a ryujin run.
*/
int main(int argc, char *argv[]){

  const std::string prm_name = argv[1];
  const std::string restart_fname = argv[2];
  const double tstart = std::stof(argv[3]);
  const double tstop  = std::stof(argv[4]);
  const unsigned int refinement = std::stoul(argv[5]);

  std::cout << "Restarting computation with file " << restart_fname << "\nending at time t in [" << tstart << ", " <<  tstop << "]." << std::endl;

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);  //create objects
  const MPI_Comm comm_world = MPI_COMM_WORLD;
  my_App app(comm_world, comm_world, {refinement});
  std::cout << "Initializing with prm = " + prm_name << std::endl;
  
  app.initialize(prm_name);

  /**
   * Deifne the postprocess lambda
   */
    const auto calculate_drag_and_lift = [&](const ryujin::TimeLoop<ryujin::Euler::Description, 2, NUMBER>::vector_type U, double t){
    using scalar_type = dealii::LinearAlgebra::distributed::Vector<NUMBER>;
    unsigned int dim = 2;
    const auto hyperbolic_system_view = app.levels[0]->hyperbolic_system->template view<2/*dim*/, NUMBER>();
    // first, set up the finite element, the data, and the facevalues
    
    const int n_q_points = app.levels[0]->offline_data->discretization().quadrature_1d().size();
    std::vector<double> pressure_values(n_q_points);
    scalar_type density, pressure, energy_density;
    std::vector<scalar_type> momentum(dim);
    // initialize partitions
    density.reinit(app.levels[0]->offline_data->scalar_partitioner(), app.levels[0]->level_comm_x);
    pressure.reinit(app.levels[0]->offline_data->scalar_partitioner(), app.levels[0]->level_comm_x);
    energy_density.reinit(app.levels[0]->offline_data->scalar_partitioner(), app.levels[0]->level_comm_x);
    for (unsigned int c = 0; c < dim; c++)
      momentum.at(c).reinit(app.levels[0]->offline_data->scalar_partitioner(),
                            app.levels[0]->level_comm_x);
    dealii::Tensor<1, 2/*dim*/> normal_vector;
    dealii::SymmetricTensor<2, 2/*dim*/> fluid_stress;
    dealii::SymmetricTensor<2, 2/*dim*/> fluid_pressure;
    dealii::Tensor<1, 2/*dim*/> forces;
    dealii::FEFaceValues<2/*dim*/> fe_face_values(
        app.levels[0]->offline_data->discretization().finite_element()/*FE_Q<dim>*/,
        app.levels[0]->offline_data->discretization().quadrature_1d()/*QGauss<dim-1*/,
        dealii::update_values | dealii::update_quadrature_points |
            dealii::update_gradients | dealii::update_JxW_values |
            dealii::update_normal_vectors); // the face values
    // Create vectors that store the locally owned parts on every process
    U.extract_component(density, 0);        // extract density
    U.extract_component(pressure, dim + 1); // extract density
    // extract momentum, and convert to velocity
    for (unsigned int c = 0; c < dim; c++) {
      int comp =
          c + 1; // momentum is stored in positions [1,...,dim], so add one to c
      U.extract_component(momentum.at(c), comp);
    }
    // extract energy
    U.extract_component(energy_density, dim + 1);
    // convert E to pressure
    for (unsigned int k = 0; k < app.levels[0]->offline_data->n_locally_owned(); k++) {
      // calculate momentum norm squared
      const double &E = energy_density.local_element(k);
      const double &rho = density.local_element(k);
      double m_square = 0;
      for (unsigned int d = 0; d < dim; d++)
        m_square += std::pow(momentum.at(d).local_element(k), 2);
      // pressure = (gamma-1)*internal_energy
      pressure.local_element(k) = (hyperbolic_system_view.gamma() - 1.0) * (E - 0.5 * m_square / rho);
    }
    density.update_ghost_values();
    pressure.update_ghost_values();
    for (auto mom : momentum)
      mom.update_ghost_values();
    double drag = 0.;
    double lift = 0.;
    for (const auto &cell : app.levels[0]->offline_data->dof_handler().active_cell_iterators()) {
      if (cell->is_locally_owned()) {
        for (unsigned int face = 0; face < cell->n_faces(); ++face) {
          if (cell->face(face)->at_boundary() &&
              cell->face(face)->boundary_id() == ryujin::Boundary::object) {
            // if on circle, we do the calculation
            // first, find if the face center is on the circle
              fe_face_values.reinit(cell, face);
              // pressure values
              fe_face_values.get_function_values(pressure, pressure_values);
              // now, loop over quadrature points calculating their contribution
              // to the forces acting on the face
              for (int q = 0; q < n_q_points; ++q) {
                normal_vector = -fe_face_values.normal_vector(q);
                // form the contributions from pressure
                for (unsigned int d = 0; d < dim; ++d)
                  fluid_pressure[d][d] = pressure_values[q];
                fluid_stress = -fluid_pressure; // for the euler equations, the
                                                // only contribution to stresses
                                                // comes from pressure
                forces = fluid_stress * normal_vector * fe_face_values.JxW(q);
                // the drag is in the x direction, the lift is in the y
                // direction but FIXME: does this hold true in higher dimension?
                // look below for this
                drag += forces[0];
                lift += forces[1];
              } // loop over q points
          }     // if cell face is at boundary && on the object
        }       // face loop
      }         // locally_owned cells
    }           // cell loop
    // now, sum the values across all processes.
    lift = dealii::Utilities::MPI::sum(lift, app.levels[0]->level_comm_x);
    drag = dealii::Utilities::MPI::sum(drag, app.levels[0]->level_comm_x);
    forces[0] = drag;
    forces[1] = lift;
    
    std::string message = "t: " + std::to_string(t) + " drag: " + std::to_string(drag) + " lift: " + std::to_string(lift);
    if(dealii::Utilities::MPI::this_mpi_process(app.levels[0]->level_comm_x) == 0)
      std::cout << message << std::endl;
  };

  // Set up data.
  ryujin::TimeLoop<ryujin::Euler::Description, 2, NUMBER>::vector_type U;
  U.reinit_with_scalar_partitioner(app.levels[0]->offline_data->scalar_partitioner());

  // Initialize data needs to be at t = 0.
  Assert(std::fabs(tstart) < 1.e-6,
    dealii::StandardExceptions::ExcMessage("tstart needs to be zero for this executble."
      "Here, tstart=" + std::to_string(tstart)));
  U = app.levels[0]->initial_values->interpolate(/*0*/);
  U.update_ghost_values();

  app.time_loops[0]->change_base_name(restart_fname);
  //now that we have the data, we call the run function
  app.time_loops[0]->run_with_initial_data(U, tstop, tstart, /*mgrit_specified_printing*/true, calculate_drag_and_lift);

  return 1;
}