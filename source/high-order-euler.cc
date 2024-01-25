/*
 * Using ryujin for the higher order Euler simulation of
 * a mach 3 flow around a cylinder.
 */

//includes
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cassert>
#include <sys/stat.h>
#include <sys/types.h>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include <string>
#include <utility>


//MPI
#include <deal.II/base/mpi.h>

//ryujin includes
#include "hyperbolic_module.h"
#include "offline_data.h"
#include "geometry_cylinder.h"
#include "discretization.h"
#include "hyperbolic_system.h"
#include "euler/parabolic_system.h"
#include "time_loop.h"
#include "euler/description.h"
#include "initial_values.h"
#include "offline_data.h"
#include "parabolic_module.h"
#include "postprocessor.h"
#include "quantities.h"
#include "time_integrator.h"
#include "vtu_output.h"
#include "convenience_macros.h"
#include "time_loop_mgrit.h"
#include "time_loop_mgrit.template.h"

//deal.II includes
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/base/tensor.h>

//xbraid include
#include <braid.h>
#include <braid_test.h>

//mgrit includes
#include "level_structures.h"

/*
 * Author: Jerett Cherry, Colorado State University
 */

/**
 *   \brief Contains the implementation of the mandatory X-Braid functions
 *
 *  X-Braid mandates several functions in order to drive the solution.
 *  This file contains the implementation of said mandatory functions.
 *  See the X-Braid documentation for more information.
 *  There are several functions that are optional in X-Braid that may
 *  or may not be implemented in here.
 */

// This preprocessor macro is used on function arguments
// that are not used in the function. It is used to
// suppress compiler warnings.
#define UNUSED(x) (void)(x)

/**
 * print the local elements and ghost elements of a partitioner
*/
void print_partition(const dealii::Utilities::MPI::Partitioner & partition, const MPI_Comm comm = MPI_COMM_WORLD)
{
  for(unsigned int proc = 0; proc < dealii::Utilities::MPI::n_mpi_processes(comm); proc++)
  {
    if(dealii::Utilities::MPI::this_mpi_process(comm)==proc)
    {
      std::cout << "Proc:" << proc <<std::endl;
      std::cout <<"Local range___________________" << std::endl;
      partition.locally_owned_range().print(std::cout);
      std::cout << "ghost elements" << std::endl;
      partition.ghost_indices().print(std::cout);
    }
  }
}

// This struct contains all data that changes with time. For now
// this is just the solution data. When doing AMR this should
// probably include the triangulization, the sparsity patter,
// constraints, etc.
/**
 * \brief Struct that contains a ryujin::TimeLoop::vector_type of size problem_dimension*n_dofs.
 */
typedef struct _braid_Vector_struct
{
  ryujin::TimeLoop<ryujin::Euler::Description, 2, NUMBER>::vector_type U;
} my_Vector;


// This struct contains all the data that is unchanging with time.
/**
 * The app structure that xbraid passes around to all its functions. Contains all
 * level specific information that we will query.
 */
typedef struct _braid_App_struct : public dealii::ParameterAcceptor
{
    using Description = ryujin::Euler::Description;
    using Number = NUMBER;
    using LevelType
        = std::shared_ptr<ryujin::mgrit::LevelStructures<Description, 2, Number>>;
    using TimeLoopType
        = std::shared_ptr<ryujin::mgrit::TimeLoopMgrit<Description,2,Number>>;

  public:
    using HyperbolicSystemView =
        typename Description::HyperbolicSystem::template View<2, Number>;
    static constexpr unsigned int problem_dimension =
        HyperbolicSystemView::problem_dimension;
    static constexpr unsigned int n_precomputed_values =
        HyperbolicSystemView::n_precomputed_values;
    using scalar_type = ryujin::OfflineData<2, Number>::scalar_type;
    using vector_type = ryujin::MultiComponentVector<Number, problem_dimension>;//TODO: determine if I need these typenames at all in app;
    using precomputed_type = ryujin::MultiComponentVector<Number, n_precomputed_values>;

    const MPI_Comm comm_x, comm_t; //todo: can I make this const? uninitialized communicators now.
    std::vector<LevelType> levels; //instantiation
    std::vector<unsigned int> refinement_levels;
    std::vector<TimeLoopType> time_loops;
    int finest_index, coarsest_index;
    unsigned int n_fine_dofs;
    unsigned int n_locally_owned_dofs;
    bool print_solution = false;
    int num_time;
    double global_tstart;
    double global_tstop;
    braid_Int cfactor;//FIXME: decide what types here? ryujin types or braid types? what about doubles??
    int max_iter;

    //a pointer to store a vector which represents the time average of all time points (after 0)
    // my_Vector* time_avg;

    _braid_App_struct(const MPI_Comm comm_x,
                      const MPI_Comm comm_t,
                      const std::vector<unsigned int> a_refinement_levels)
    // _braid_App_struct(const MPI_Comm comm_world)
    : ParameterAcceptor("/App"),
      comm_x(comm_x),
      comm_t(comm_t),
      levels(a_refinement_levels.size()),
      refinement_levels(a_refinement_levels),
      time_loops(a_refinement_levels.size()),
      finest_index(0)//for XBRAID, the finest level is always 0.
    {
      // refinement_levels = {5,2,1};
      // add_parameter("mgrit refinements",
      //               refinement_levels,
      //               "Vector of levels of global mesh refinement where "
      //               "each MGRIT level will work on.");
      coarsest_index = refinement_levels.size()-1;

      print_solution = false;
      add_parameter("print_solution",
                    print_solution,
                    "Optional to print the solution in Init and Access.");
      num_time = 4;
      add_parameter("Time Bricks",
                    num_time,
                    "Number of time bricks total.");
      global_tstart = 0.0;
      add_parameter("Start Time",
                    global_tstart);//TODO: add a description, these describe the global time domain.

      global_tstop = 5.0;
      add_parameter("Stop Time",
                    global_tstop);
      
      cfactor = 2; //The default coarsening from level to level is 2, which matches that of XBRAID.
      add_parameter("cfactor",
                    cfactor,
                    "The coarsening factor between time levels.");
      max_iter = num_time;//In theory, mgrit should converge after the number of cycles equal to the number of time points it has.
      add_parameter("max_iter", 
                    max_iter,
                    "The maximum number of MGRIT iterations.");
    };

    // Calls ParameterAcceptor::initialize(prm_name), this will initialize all the data structures.
    void initialize(const std::string prm_name)
    {
      //Reorder refinement levels in descending order of refinement,
      //this matches the fact that Xbraid has the finest level of MG
      //as 0. I.E. the most refined data is accessed with refinement_levels[0]
      std::sort(refinement_levels.rbegin(), refinement_levels.rend());
      //TODO: need to make a way to remove duplicates, or at least warn user
      //that duplicate refinement levels are inefficient.

      create_mg_levels();

      //now that levels are all created, we parse the parameter file.
      dealii::ParameterAcceptor::initialize(prm_name);

      //all parameters defined, we can now call all objects prepare function.
      prepare_mg_objects();
      initialized = true;//now the user can access data in app. TODO: implement a check for getter functions.
    }

  private:

    //prepares all created objects at each level
    void prepare_mg_objects()
    {
      for(unsigned int lvl = 0; lvl < refinement_levels.size(); lvl++)
      {
        if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0)
        {
          std::cout << "[INFO] Preparing Structures in App at level "
                    << refinement_levels[lvl] << std::endl;
        }
        levels[lvl]->prepare();
      }
      //set the last variables in app.
      n_fine_dofs = levels[0]->offline_data->dof_handler().n_dofs();
      n_locally_owned_dofs = levels[0]->offline_data->n_locally_owned();
    }

    //declares all objects at all levels
    void create_mg_levels()
    {
      for(unsigned int i=0; i<refinement_levels.size(); i++)
      {
        if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0)
        {
          std::cout << "[INFO] Setting Structures in App at level "
              << refinement_levels[i] << std::endl;
        }
        //TODO: determine if I should just make a time loop object for each level and using only this.
        // i.e. does app really ned to know all the level structures info?
        levels[i] = std::make_shared<ryujin::mgrit::LevelStructures<Description, 2, Number>>(comm_x, refinement_levels[i]);
        time_loops[i] = std::make_shared<ryujin::mgrit::TimeLoopMgrit<Description,2,Number>>(comm_x, *(levels[i]), 
                                                                                             0/*initial time is irrelevant*/,
                                                                                             0/*final time is irrelevant*/);
      }
    }

    bool initialized = false;//FIXME: unused, delete?

    // void set_time_averaged_pointer(my_Vector& a_time_avg) {time_avg = a_time_avg;};
} my_App;

/**
 * @brief Asserts that no value in the solution vector is nan. 
 * 
 *  This should not be used on large vectors as it loops through all elements.
 * 
 * @param app - The app which describes all levels
 * @param U - The vector to test
*/

void no_nans(const my_App &app, const my_Vector &U, const std::string caller)
{
  UNUSED(app);
  for (unsigned int i=0; i < U.U.size(); i++)
  {
    // std::cout << "Testing nans: " << i << " isNAN: " << std::isnan(U.U[i]) <<  std::endl;
    Assert(!std::isnan(U.U[i]), ExcMessage("Nan in solution after " + caller));
  }
}

/**
 * @brief Calculates drag and lift at a given my_Vector and app.
 * 
 *   This function calculates the drag and lift of a vector, and returns the pair of them.
 *   ONLY works if dim = 2.
 * 
*/
// template<int dim>//TODO: template this?
dealii::Tensor<1,2/*dim*/> calculate_drag_and_lift(const my_Vector& u, const my_App& app, const NUMBER t, unsigned int dim = 2)
{
  if (dim == 2) {
    using scalar_type = dealii::LinearAlgebra::distributed::Vector<NUMBER>;
    
    UNUSED(t);

    const auto offline_data = app.levels[app.finest_index]->offline_data;
    const auto mpi_communicator = app.comm_x;
    const auto hyperbolic_system_view = app.levels[app.finest_index]->hyperbolic_system->template view<2/*dim*/, NUMBER>();
    // first, set up the finite element, the data, and the facevalues
    // const dealii::FiniteElement<2,2> fe = app.levels[app.finest_index]->offline_data->discretization().finite_element(); // the finite element
    // const int degree = app.levels[app.finest_index]->offline_data->discretization().finite_element().degree;
    // dealii::QGauss<1> face_quadrature_formula = app.levels[app.finest_index]->offline_data->discretization().quadrature_1d();
    const int n_q_points = app.levels[app.finest_index]->offline_data->discretization().quadrature_1d().size();

    std::vector<double> pressure_values(n_q_points);

    scalar_type density, pressure;
    std::vector<scalar_type> momentum(dim);

    // initialize partitions
    density.reinit(offline_data->scalar_partitioner(), mpi_communicator);
    pressure.reinit(offline_data->scalar_partitioner(), mpi_communicator);
    for (unsigned int c = 0; c < dim; c++)
      momentum.at(c).reinit(offline_data->scalar_partitioner(),
                            mpi_communicator);

    dealii::Tensor<1, 2/*dim*/> normal_vector;
    dealii::SymmetricTensor<2, 2/*dim*/> fluid_stress;
    dealii::SymmetricTensor<2, 2/*dim*/> fluid_pressure;
    dealii::Tensor<1, 2/*dim*/> forces;

    dealii::FEFaceValues<2/*dim*/> fe_face_values(
        app.levels[app.finest_index]->offline_data->discretization().finite_element()/*FE_Q<dim>*/,
        app.levels[app.finest_index]->offline_data->discretization().quadrature_1d()/*QGauss<dim-1*/,
        dealii::update_values | dealii::update_quadrature_points |
            dealii::update_gradients | dealii::update_JxW_values |
            dealii::update_normal_vectors); // the face values

    // Create vectors that store the locally owned parts on every process
    u.U.extract_component(density, 0);        // extract density
    u.U.extract_component(pressure, dim + 1); // extract density

    // extract momentum, and convert to velocity
    for (unsigned int c = 0; c < dim; c++) {
      int comp =
          c + 1; // momentum is stored in positions [1,...,dim], so add one to c
      u.U.extract_component(momentum.at(c), comp);
    }

    // extract energy
    u.U.extract_component(pressure, dim + 1);

    // convert E to pressure
    for (unsigned int k = 0; k < offline_data->n_locally_owned(); k++) {

      // calculate momentum norm squared
      const double &E = pressure.local_element(k);
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

    for (const auto &cell : offline_data->dof_handler().active_cell_iterators()) {
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
    lift = dealii::Utilities::MPI::sum(lift, mpi_communicator);
    drag = dealii::Utilities::MPI::sum(drag, mpi_communicator);

    forces[0] = drag;
    forces[1] = lift;
    return forces;

  } /* dim is 2, this code only works for dim=2, unfortunately.*/ else {
    Assert(false, 
      ExcMessage("You are trying to call calculate_drag_and_lift,"
      " which only works for dim=2. you called with dim = " + std::to_string(dim)));
    dealii::Tensor<1,2/*dim*/> forces;
    return forces;
  }
}

/**
 * @brief a ryujin::MulticomponentVector<double, 4> at a time t.
 * @param v - The vector to be printed.
 * @param app - the my_App object which stores the timeloop which does the printing
 * @param t - the time t at which the vector is printed fixme: does this need to be here? could put it in the fname
 * @param level - the level from which this vector comes. fixme: this should really not be a separate thing. For example, one could call this for a vector from level 0, but ask it to print level 8. this is a problem
 * @param 
*/
void print_solution(ryujin::MultiComponentVector<double, 4> &v,
                    const braid_App &app,
                    const double t = 0,
                    const unsigned int level = 0,
                    const std::string fname = "./test-output")
{
  std::cout << "printing solution" << std::endl;
  const auto time_loop = app->time_loops[level];
  time_loop->output(v, fname + std::to_string(t), t /*current time*/, 1/*cycle*/);
}

/**Temporary function to print out cells that interpolate between levels will call.*/
template<int dim, int spacedim>
void print_cells_on_dofs(dealii::DoFHandler<dim,spacedim>& dof)
{
  typename dealii::DoFHandler<dim,spacedim>::cell_iterator cell = dof.begin();
  const typename dealii::DoFHandler<dim,spacedim>::cell_iterator endcell = dof.end();

  const dealii::types::subdomain_id tria_id = dof.get_triangulation().locally_owned_subdomain();
}

/**
 * @brief This function is used to interpolate a vector (from_V) on a certain mesh
 *        to a nother interpolated solution on another mesh.
 *
 * @param  to_v - the vector V to which we are interpolating the solution
 * @param  to_level - the mg level we are interpolating to
 * @param  from_V - the vector storing the solution we wish to interpolate
 * @param  from_level - the mg level which currently stores the solution
 * @param  app - the user defined app which stores needed information about the mg levels
 *
 * NOTE: this function only works if the vectors to_v and from_v have an extract_component method implemented.
 * and the method insert_component;
 */
void interpolate_between_levels(my_Vector& to_v,
                               const unsigned int to_level,
                               const my_Vector& from_v,
                               const unsigned int from_level,
                               const braid_App& app)
{
  Assert((to_v.U.size() == app->levels[to_level]->offline_data->dof_handler().n_dofs()*app->problem_dimension)
      , ExcMessage("Trying to interpolate to a vector and level where the n_dofs do not match will not work."));
  using scalar_type = ryujin::OfflineData<2,NUMBER>::scalar_type;
  scalar_type from_component,to_component;

  const unsigned int problem_dimension = app->problem_dimension;
  const auto &from_partitioner = app->levels[from_level]->offline_data->scalar_partitioner();
  const auto &to_partitioner = app->levels[to_level]->offline_data->scalar_partitioner();
  const auto &comm = app->comm_x;

  // print_partition(*from_partitioner);
  // print_partition(*to_partitioner);
  //reinit the components to match the correct info.
  from_component.reinit(from_partitioner,comm);
  to_component.reinit(to_partitioner,comm);

  for(unsigned int comp=0; comp<problem_dimension; comp++)
  {
    // extract component
    from_v.U.extract_component(from_component, comp);
    // interpolate this into the to_component
    dealii::VectorTools::interpolate_to_different_mesh(
        app->levels[from_level]->offline_data->dof_handler(),
        from_component,
        app->levels[to_level]->offline_data->dof_handler(),
        to_component);
    // place component
    to_v.U.insert_component(to_component, comp);
  }
  to_v.U.update_ghost_values();
  // no_nans(*app, to_v, "interpolateBetweenLevels");//remove
}

///This function reinits a vector to the specified level, making sure that the partition matches that of the level.
void reinit_to_level(my_Vector* u, braid_App& app, const unsigned int level) {
  Assert(app->levels.size()>level, ExcMessage("The level being reinitialized does not exist."));
  u->U.reinit_with_scalar_partitioner(
      app->levels[level]->offline_data->scalar_partitioner());
  u->U.update_ghost_values();//TODO: is this neccessary?
}

/**
 * @brief my_Step - Interpolates u to the correct level, calls run_with_initial_data and updates u.
 *
 * @param app - The braid app struct
 * @param ustop - The solution data at the end of this time step
 * @param fstop - RHS data (such as forcing function?)
 * @param u - The solution data at the beginning of this time step
 * @param status - Status structure that contains various info of this time
 *
 * @return Success (0) or failure (1)
 **/
int my_Step(braid_App        app,
            braid_Vector     ustop,
            braid_Vector     fstop,
            braid_Vector     u/*u at the finest spatial level*/,
            braid_StepStatus status)
{
  //this variable is used for writing data to
  //different files during the parallel computations.
  //is passed to run_with_initial_data
  static unsigned int num_step_calls = 0;

  //grab the MG level for this step
  int level;
  braid_StepStatusGetLevel(status, &level);

  //use a macro to get rid of some unused variables to avoid -Wall messages
  UNUSED(ustop);
  UNUSED(fstop);
  //grab the start time and end time
  double tstart;
  double tstop;
  braid_StepStatusGetTstartTstop(status, &tstart, &tstop);

  if (1/*this was a mpi comm id check before FIXME*/) 
  {
    std::cout << "[INFO] Stepping on level: " + std::to_string(level)+ "\non interval: [" +std::to_string(tstart)+ ", "+std::to_string(tstop)+ "]\n"
      + "total step call number " +std::to_string(num_step_calls) << std::endl;
  }

  //translate the fine level u coming in to the coarse level
  //this uses a function from DEALII interpolate to different mesh

  //new, coarse vector
  my_Vector u_to_step;
  reinit_to_level(&u_to_step, app, level);

  //interpolate between levels, put data from u (fine level) onto the u_to_step (coarse level)
  interpolate_between_levels(u_to_step, level, *u, 0, app);
  //step the function on this level
  app->time_loops[level]->run_with_initial_data(u_to_step.U, tstop, tstart);

  //interpolate this back to the fine level
  interpolate_between_levels(*u,0,u_to_step,level, app);
  num_step_calls++;
  //done.
  return 0;
};

/**
 * @brief my_Init - Initializes a solution data at the given time
 * For this, the init function initializes each time point with a coarsest grid calculation of the solution.
 *
 * @param app - The braid app struct containing user data
 * @param t - Time at which the solution is initialized
 * @param u_ptr - The solution data that needs to be filled
 *
 * @return Success (0) or failure (1)
 **/
int
my_Init(braid_App     app,
        double        t,
        braid_Vector *u_ptr)
{
  std::cout << "[INFO] Initializing XBraid vectors at t=" << t << std::endl;

  // We first define a coarse vector, at the coarsest level, which will be stepped, then restricted down to the fine level
  // and interpolate the fine initial state into the coarse vector, then interpolates it up to the coarse level and steps.
  my_Vector *u = new(my_Vector);
  my_Vector *temp_coarse = new(my_Vector);
  reinit_to_level(u, app, app->finest_index);//this is the u that we will start each time brick with.
  reinit_to_level(temp_coarse, app, app->coarsest_index);//coarse on the coarses level.
  u->U = app->levels[app->finest_index]->initial_values->interpolate(0);//sets up U data at t=0;
  temp_coarse->U = app->levels[app->coarsest_index]->initial_values->interpolate(0);
  
  std::string str = "initialized_at_t=" + std::to_string(t);
  // If T is not zero, we step on the coarsest level until we are done. Otherwise we have no need to step any because
  // the assumtion is that T=0
  //TODO: implicit assumption that T>0 always here except for T=0.
  if(std::fabs(t) > 0)
  {
    //interpolate the initial conditions up to the coarsest mesh
    interpolate_between_levels(*temp_coarse, app->coarsest_index, *u, app->finest_index, app);
    //steps to the correct end time on the coarse level to end time t
    app->time_loops[app->coarsest_index]->run_with_initial_data(temp_coarse->U,t);
    if(app->print_solution)
      print_solution(temp_coarse->U, app, t, app->coarsest_index, str);

    interpolate_between_levels(*u, app->finest_index, *temp_coarse, app->coarsest_index, app);
  }
  //delete the temporary coarse U. 
  delete temp_coarse;

  //reassign pointer XBraid will use
  *u_ptr = u;
  return 0;
}


/**
 * @brief my_Clone - Clones a vector into a new vector
 *
 * @param app - The braid app struct containing user data
 * @param u - The existing vector containing data
 * @param v_ptr - The empty vector that needs to be filled
 *
 * @return Success (0) or failure (1)
 **/
int
my_Clone(braid_App     app,
         braid_Vector  u,
         braid_Vector *v_ptr)
{
  if (dealii::Utilities::MPI::this_mpi_process(app->comm_t) == 0)
  {
    std::cout << "[INFO] Cloning XBraid vectors" << std::endl;
  }

  my_Vector *v = new(my_Vector);
  //all vectors are 'fine level' vectors
  reinit_to_level(v,app,0);
  v->U.equ(1, u->U);

  *v_ptr = v;

  // no_nans(*app, *v, "clone");//remove

  return 0;
}


/**
 * @brief my_Free - Deletes a vector
 *
 * @param app - The braid app struct containing user data
 * @param u - The vector that needs to be deleted
 *
 * @return Success (0) or failure (1)
 **/
int
my_Free(braid_App    app,
        braid_Vector u)
{
  UNUSED(app);
  if (dealii::Utilities::MPI::this_mpi_process(app->comm_t) == 0)
  {
    std::cout << "[INFO] Freeing XBraid vectors" << std::endl;
  }

  delete u;

  return 0;
}


/**
 * @brief my_Sum - Sums two vectors in an AXPY operation sums on the finer of the two levels.
 * The operation is y = alpha*x + beta*y
 *
 * @param app - The braid app struct containing user data
 * @param alpha - The coefficient in front of x
 * @param x - A vector that is multiplied by alpha then added to y
 * @param beta - The coefficient of y
 * @param y - A vector that is multiplied by beta then summed with x
 *
 * @return Success (0) or failure (1)
 **/
int
my_Sum(braid_App app,
       double alpha,
       braid_Vector x,
       double beta,
       braid_Vector y)
{
  UNUSED(app);

  if (dealii::Utilities::MPI::this_mpi_process(app->comm_t) == 0)
  {
    std::cout << "[INFO] Summing XBraid vectors" << std::endl;
  }

  y->U.sadd(beta, alpha, x->U);

  return 0;
}

/**
 *  \brief Returns the spatial norm of the provided vector
 *
 *  Calculates and returns the spatial norm of the provided vector.
 *  Interestingly enough, X-Braid does not specify a particular norm.
 *  to keep things simple, we implement the Euclidean norm.
 *
 *  \param app - The braid app struct containing user data
 *  \param u - The vector we need to take the norm of
 *  \param norm_ptr - Pointer to the norm that was calculated, need to modify this
 *  \return Success (0) or failure (1)
 */
int
my_SpatialNorm(braid_App     app,
               braid_Vector  u,
               double       *norm_ptr)
{
  UNUSED(app);
  if (dealii::Utilities::MPI::this_mpi_process(app->comm_t) == 0)
  {
    std::cout << "[INFO] Calculating XBraid vector spatial norm" << std::endl;
  }

  *norm_ptr = u->U.l2_norm();

  return 0;
}

/**
 *  \brief Allows the user to output details
 *
 *  The Access function is called at various points to allow the user to output
 *  information to the screen or to files.
 *  The astatus parameter provides various information about the simulation,
 *  see the XBraid documentation for details on what information you can get.
 *  Example information is what the current timestep number and current time is.
 *  If the access level (in parallel_in_time.cc) is set to 0, this function is
 *  never called.
 *  If the access level is set to 1, the function is called after the last
 *  XBraid cycle.
 *  If the access level is set to 2, it is called every XBraid cycle.
 *
 *  \param app - The braid app struct containing user data
 *  \param u - The vector containing the data at the status provided
 *  \param astatus - The Braid status structure
 *  \return Success (0) or failure (1)
 */
int
my_Access(braid_App          app,
          braid_Vector       u,
          braid_AccessStatus astatus)
{
  if (dealii::Utilities::MPI::this_mpi_process(app->comm_t) == 0)
  {
    std::cout << "[INFO] Access Called" << std::endl;
  }
  static int mgCycle = 0;
  double t = 0;

  //state what iteration we are on, and what time t we are at.
  braid_AccessStatusGetIter(astatus, &mgCycle);
  braid_AccessStatusGetT(astatus, &t);

  if (dealii::Utilities::MPI::this_mpi_process(app->comm_t) == 0)
  {
    std::cout << "Cycles done: " << mgCycle << std::endl;
  }

  std::string fname = "./cycle" + std::to_string(mgCycle);
  if(app->print_solution)
    print_solution(u->U, app, t, 0/*level, always needs to be zero, to be fixed*/, fname);

  //calculate drag and lift for this solution on level 0
  //TODO: insert drag and lift calculation call.
  dealii::Tensor<1,2> forces = calculate_drag_and_lift(*u,*app,t,2/*dim*/);
  std::cout << "cycle." + std::to_string(mgCycle) + " drag." +std::to_string(forces[0]) 
              + " lift." + std::to_string(forces[1]) + " time." +std::to_string(t) << std::endl;

  return 0;
}

/**
 *  \brief Calculates the size of a buffer for MPI data transfer
 *
 *  Calculates the size of the buffer that is needed to transfer
 *  a solution vector to another processor.
 *  The bstatus parameter provides various information on the
 *  simulation, see the XBraid documentation for all possible
 *  fields.
 *
 *  \param app - The braid app struct containing user data
 *  \param size_ptr A pointer to the calculated size
 *  \param bstatus The XBraid status structure
 *  \return Success (0) or failure (1)
 */
int
my_BufSize(braid_App           app,
           int                 *size_ptr,
           braid_BufferStatus  bstatus)
{
  if (dealii::Utilities::MPI::this_mpi_process(app->comm_t) == 0)
  {
    std::cout << "[INFO] Buf_size Called" << std::endl;
  }

  //TODO: answer question about what the buffer size whould be, i think it should be
  //problem_dimension*number of spatial nodes. But there is some question in my mind about
  //if the MPI communication is from this Time Brick (which owns a distributed vector on a few
  //processors) or among the spatial processors. I suspect the former.
  UNUSED(bstatus);

  //no vector can be bigger than this, so we are very conservative.
  int size = app->n_fine_dofs * app->problem_dimension;
  *size_ptr = (size+1)*sizeof(NUMBER);//+1 is for the size of the buffers being stored in the first component.
  std::cout << "Size in bytes of the NUMBER: " << sizeof(NUMBER) << std::endl;
  std::cout << "Problem_dimension: " << app->problem_dimension << " n_dofs: " << app->n_fine_dofs << std::endl;
  std::cout << "buf_size: " << *size_ptr << std::endl;

  return 0;
}

/**
 *  @brief Linearizes a vector to be sent to another processor
 *
 *  Linearizes (packs) a data buffer with the contents of
 *  some solution state u.
 *
 *  @param app - The braid app struct containing user data
 *  @param u The vector that must be packed into buffer
 *  @param buffer The buffer that must be filled with u
 *  @param bstatus The XBraid status structure
 *  @return Success (0) or failure (1)
 */ 
int
my_BufPack(braid_App           app,
           braid_Vector        u,
           void               *buffer,
           braid_BufferStatus  bstatus)
{
  if (dealii::Utilities::MPI::this_mpi_process(app->comm_t) == 0)
  {
    std::cout << "[INFO] BufPack Called" << std::endl;
  }

  const int problem_dimension = app->problem_dimension;
  NUMBER *dbuffer = (NUMBER*)buffer;
  unsigned int n_locally_owned = app->n_locally_owned_dofs;//number of dofs at finest level
  unsigned int buf_size = n_locally_owned * app->problem_dimension;
  dbuffer[0] = buf_size + 1;//buffer + size
  dealii::Tensor<1, problem_dimension, NUMBER> temp_tensor;
  // dealii::Tensor<1,problem_dimension, dealii::VectorizedArray<NUMBER>> temp_tensor_vectorized;
  for(unsigned int node=0; node < n_locally_owned; node++)
  {
    //extract tensor at this node
    temp_tensor = u->U.get_tensor(node);//TODO:FIXME: test linear method for speed...
    for (unsigned int component = 0; component < problem_dimension; ++component) {
      Assert(buf_size >= (node + component),
             ExcMessage("In my_BufPack, the size of node + component is "
                        "greater than the buff_size (the expected size of the vector)."));
      dbuffer[problem_dimension*(node) + component + 1] = u->U.local_element(problem_dimension*node + component);
      Assert(!std::isnan(u->U.local_element(problem_dimension*node + component)),
        ExcMessage("The vector you are trying to pack has a NaN in it at component " + std::to_string(problem_dimension*node + component)));
    }
  }
  braid_BufferStatusSetSize(bstatus, (buf_size+1)*sizeof(NUMBER));//set the number of bytes stored in this buffer (TODO: this is off since the dbuffer[0] is a integer.)
  return 0;
}

/**
 *  \brief Unpacks a vector that was sent from another processor
 *
 *  Unpacks a linear data buffer into the vector pointed to by
 *  u_ptr.
 *
 *  \param app - The braid app struct containing user data
 *  \param buffer The buffer that must be unpacked
 *  \param u_ptr The pointer to the vector that is filled
 *  \param bstatus The XBraid status structure
 *  \return Success (0) or failure (1)
 */
int
my_BufUnpack(braid_App           app,
             void               *buffer,
             braid_Vector       *u_ptr,
             braid_BufferStatus  bstatus)
{
  if (dealii::Utilities::MPI::this_mpi_process(app->comm_t) == 0)
  {
    std::cout << "[INFO] BufUnpack Called" << std::endl;
  }

  NUMBER *dbuffer = (NUMBER*)buffer;
  int buf_size = static_cast<int>(dbuffer[0]);//TODO: is this dangerous?
  const int problem_dimension = app->problem_dimension;

  // The vector should be size (dim + 2) X n_dofs at finest level.
  my_Vector *u = NULL; // the vector we will pack the info into
  u = new(my_Vector);//TODO: where does this get deleted? Probably wherever owns the u_ptr.
  reinit_to_level(u, app, app->finest_index);//each U is at the finest level.
  dealii::Tensor<1, problem_dimension, NUMBER> temp_tensor;

  //unpack the sent data into the right level
  for(unsigned int node=0; node < app->n_locally_owned_dofs; node++)
  {
    //get tensor at node.
    for(unsigned int component = 0; component < app->problem_dimension; ++component)
    {
      // temp_tensor[component] = dbuffer[app->problem_dimension*node + component + 1];//+1 because buffer_size = n_dof + 1
      u->U.local_element(problem_dimension*node + component) = dbuffer[app->problem_dimension*node + component + 1];//test for speed.
      Assert(node + component +1 <= buf_size,
          ExcMessage("somehow, you are exceeding the buffer size as you unpack"));
    }
    //insert tensor at node
    // u->U.write_tensor(temp_tensor, node);//TODO:FIXME: try linear method?
  }

  *u_ptr = u;//modify the u_ptr does this create a memory leak as we just point this pointer somewhere else?

  return 0;
}

void test_braid_functions(my_App& app, braid_MPI_Comm comm_x = MPI_COMM_WORLD)
{
  my_Vector *V = NULL;
  V = new (my_Vector);

  // test my_Init
  my_Init(&app, 2.0, &V);

  // test my_Clone
  my_Vector *V_cloned;
  my_Clone(&app, V, &V_cloned);
  if(app.print_solution)
    print_solution(V_cloned->U,
                  &app,
                  10101 /*time set to arbitrary time identified with clone*/,
                  0);

  // test my_SpatialNorm
  double norm = 0;
  double norm_cloned = norm;
  my_SpatialNorm(&app, V_cloned, &norm_cloned);
  my_SpatialNorm(&app, V, &norm);
  std::cout << "Norm V: " << norm << std::endl;

  Assert(
      std::abs(norm - norm_cloned) < 1e-6,
      ExcMessage("The norm of V and the norm of the cloned V do not match."));
  std::cout << "Norm assertion passed." << std::endl;

  // test my_Sum
  my_Sum(&app, 1, V_cloned, 1, V);
  my_SpatialNorm(&app, V, &norm);
  std::cout << "Norm x+x: " << norm << std::endl;
  my_Sum(&app, 2, V_cloned, 0, V);
  my_SpatialNorm(&app, V, &norm);
  std::cout << "Norm 2*x: " << norm << std::endl;

  // test my_BuffSize
  int size = 0;
  my_BufSize(&app, &size, NULL);
  std::cout << "Buffer size: " << size << std::endl;

  // test my_Free TODO: what test should this do?
  my_Free(&app, V);
  my_Free(&app, V_cloned);

  // Assert((V == NULL && V_cloned == NULL),
  //        ExcMessage("The pointers are not null after free."));

    //tests for functions
  braid_TestAll(&app,
                comm_x,
                stdout,/*FILE *fp,*/
                0.0/*time to initialize*/,
                1.0,/*time step to take from what you initialize*/
                0.0,/*time step used for coarsen and refine test, since I do not need this function, it does not matter*/
                my_Init,
                my_Free,
                my_Clone,
                my_Sum,
                my_SpatialNorm, 
                my_BufSize,
                my_BufPack,
                my_BufUnpack,
                NULL,/*coarsen function*/
                NULL,/*refine function*/
                NULL,/*residual*/
                my_Step);

}

//todo: change this to a call to something similar to the main ryujin executable. problem_dispach??
int main(int argc, char *argv[])
{
  //scoped MPI object, no need to call finalize at the end.
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);  //create objects
  MPI_Comm comm_world = MPI_COMM_WORLD;//create MPI_object
  //set up app and all underlying data, initialize parameters
  //parse command line parameters, order should be file name, parameter file, px, then the mg hierarcy, i.e. list of refinement levels.
  Assert(argc > 1/*program*/ + 1/*parameter file*/ + 1/*px*/ + 1/*at least one level refinement*/,
         ExcMessage("You must provide the program with a parameter file, a number of spatial processors, "
              "and a multigrid hierarcy. Here, the number of additional parameters needed is at least:" + std::to_string(4-argc)));
  const std::string prm_name(argv[1]);// prm file
  const int px = std::stoi(argv[2]);  // number of processors to use in space
  std::vector<unsigned int> refinement_levels(argc - 3);// the vector of refinement levels are equal to the number of remaining arguments, set to argc-3, where 3 is the number of arguments needed before the mg_hierarchy
  for(int i = 3; i < argc; i++)
    refinement_levels[i-3] = std::stoi(argv[i]);

  for(const auto entry: refinement_levels)
    std::cout << entry << std::endl;
  //split the object into the number of time processors, and the number of spatial processors per time chunk.
  MPI_Comm comm_x, comm_t;
  std::cout << "px: " << px << std::endl;

  /**
   * Split WORLD into a time brick for each processor, with a specified number of processors for each to do the spatial MPI.
   * The number of time bricks is equal to NumberProcessorsOnSystem/px    //FIXME: is this true??
   */
  Assert(dealii::Utilities::MPI::n_mpi_processes(comm_world) % px == 0,
         ExcMessage(
             "You are trying to divide world into a number of spatial "
             "processors per time brick that will cause MPI to stall. The "
             "variable px needs to divide the number of processors total."));
  braid_SplitCommworld(&comm_world,
                       px /*the number of spatial processors per time brick*/,
                       &comm_x,
                       &comm_t);

  // now that we have the communicators, we can create the app, and initialize with the parameter file.
  my_App app(comm_x, comm_t, refinement_levels);
  app.initialize(prm_name);

  /* Initialize Braid */
  braid_Core core;
  double tstart = app.global_tstart;
  double tstop = app.global_tstop;
  int ntime = app.num_time;//this should in general be the number of time bricks you want. You need to ensure that px * ntime = TOTAL NUMBER PROCESSORS

  std::cout << "Start: " << tstart << " Stop: " << tstop << " # bricks: " << ntime << std::endl;

  braid_Init(comm_world,
             app.comm_t,
             tstart,
             tstop,
             ntime,
             &app,
             my_Step,
             my_Init,
             my_Clone,
             my_Free,
             my_Sum,
             my_SpatialNorm,
             my_Access,
             my_BufSize,
             my_BufPack,
             my_BufUnpack,
             &core);

  /* Define XBraid parameters
   * See -help message for descriptions */
  unsigned int max_levels = app.refinement_levels.size(); // fixme, cthis is later cast to an int.
  int nrelax = 1; // FIXME: add this and delete the UNUSED(nrelax) later.
  //      int       skip          = 0;
  double tol = 1e-2;
  int max_iter = app.max_iter;
  //      int       min_coarse    = 10;
  // int       fmg           = test_case.fmg;
  // int       scoarsen      = 0;
  // int       res           = 0;
  // int       wrapper_tests = 0;
  int print_level = 2;
  /*access_level=1 only calls my_access at end of simulation*/
  int access_level = 2;
  int use_sequential = 0; //same as XBRAID default, initial guess is from user defined init.

  UNUSED(nrelax);

  braid_SetPrintLevel(core, print_level);
  braid_SetAccessLevel(core, access_level);
  braid_SetMaxLevels(core, max_levels);
  //             braid_SetMinCoarse( core, min_coarse );
  //             braid_SetSkip(core, skip);
  //      braid_SetNRelax(core, -1, nrelax);
  braid_SetAbsTol(core, tol);
  braid_SetCFactor(core, -1, app.cfactor);
  braid_SetMaxIter(core, max_iter);
  braid_SetSeqSoln(core, use_sequential);
  std::cout << "before braid_drive\n";
  braid_Drive(core);

  // Free the memory now that we are done
  braid_Destroy(core);

}
