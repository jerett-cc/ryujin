/*
 * Using ryujin for the higher order Euler simulation of
 * a mach 3 flow around a cylinder.
 */

//includes
#include <iostream>
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
 * This function calculates the forces actind on an object in the domain (a specific boundary::id) and returns a dealii::Tensor<1,dim>.
 * In 2d, this is a Tensor<1,2> where the first component gives us the drag acting on an object.
 * 
 * \param app - The app from the MGRIT simulation storing relevant data structures.
 * \param V - The my_Vector which we 
*/

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


    const MPI_Comm comm_x, comm_t;
    std::vector<LevelType> levels; //instantiation
    std::vector<unsigned int> refinement_levels;

    std::vector<TimeLoopType> time_loops;
    int finest_index, coarsest_index;
    unsigned int n_fine_dofs;

    //a pointer to store a vector which represents the time average of all time points (after 0)
    // my_Vector* time_avg;

    _braid_App_struct(const MPI_Comm comm_x,
                      const MPI_Comm comm_t)
    : ParameterAcceptor("/MGRIT"),
      comm_x(comm_x),
      comm_t(comm_t),
      levels(1),
      time_loops(1),
      finest_index(0)//for XBRAID, the finest level is always 0.
    {
      refinement_levels = {5,2,1};
      add_parameter("mgrit refinements",
                    refinement_levels,
                    "Vector of levels of global mesh refinement where "
                    "each MGRIT level will work on.");
      coarsest_index = refinement_levels.size()-1;
      //need to set up temporary objects so that we can call ParameterAcceptor::initialize and
      //all subsections will have been defined. Then, we fill the correct information by calling
      levels[0] = std::make_shared<ryujin::mgrit::LevelStructures<Description, 2, Number>>(comm_x, 0);
      time_loops[0] = std::make_shared<ryujin::mgrit::TimeLoopMgrit<Description, 2, Number>>(comm_x, *levels[0],0,0);
      n_fine_dofs = levels[0]->offline_data->dof_handler().n_dofs();
      //default objects. their member variables will be modified when
      //parameteracceptor initialize is called, then you can call prepare.
      //FIXME: a potential bug can be introduced when a user calls prepare()
      //before they call parameterhandler::initialize, how to stop this?
    };

    //This function NEEDS to be called after parameter handler has been initialized
    //otherwise user defined refinement levels are not going to be used, only the
    //default above.
    //TODO: create a warning or a bool that warns users who try to use unprepared app.
    void prepare()
    {
      //reset the coarsest index
      coarsest_index = refinement_levels.size()-1;

      //clear the vectors
      levels.clear();
      time_loops.clear();
      //resize the vectors storing level data
      levels.resize(refinement_levels.size());
      time_loops.resize(refinement_levels.size());

      //reorder refinement levels in descending order,
      //this matches the fact that Xbraid has the finest level of MG
      //as 0.
      std::sort(refinement_levels.rbegin(), refinement_levels.rend());

      //TODO: need to make a way to remove duplicates, or at least warn user
      //that duplicate refinement levels are inefficient.

      for(unsigned int i=0; i<refinement_levels.size(); i++)
      {
        if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0)
        {
          std::cout << "[INFO] Setting up Structures in App at level "
              << refinement_levels[i] << std::endl;
        }
        //TODO: determine if I should just make a time loop object for each level and using only this.
        // i.e. does app really ned to know all the level structures info?
        levels[i] = std::make_shared<ryujin::mgrit::LevelStructures<Description, 2, Number>>(comm_x, refinement_levels[i]);
        time_loops[i] = std::make_shared<ryujin::mgrit::TimeLoopMgrit<Description,2,Number>>(comm_x, *(levels[i]), 0,0);
      }
      n_fine_dofs = levels[0]->offline_data->dof_handler().n_dofs();
    }

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
  for (unsigned int i=0; i < U.U.size(); i++)
  {
    // std::cout << "Testing nans: " << i << " isNAN: " << std::isnan(U.U[i]) <<  std::endl;
    Assert(!std::isnan(U.U[i]), ExcMessage("Nan in solution after " + caller));
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
void interpolateUBetweenLevels(my_Vector& to_v,
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

  //reinit the components to match the correct info.
  from_component.reinit(from_partitioner,comm);
  to_component.reinit(to_partitioner,comm);
  // no_nans(*app, from_v, "before interpolation");
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

  // no_nans(*app, to_v, "interpolateBetweenLevels");//remove
}

///This function reinits a vector to the specified level, making sure that the partition matches that of the level.
void reinit_to_level(my_Vector* u, braid_App& app, const unsigned int level) {
  Assert(app->levels.size()>level, ExcMessage("The level being reinitialized does not exist."));
  u->U.reinit_with_scalar_partitioner(
      app->levels[level]->offline_data->scalar_partitioner());
}

/**
 * @brief my_Step - Creates new eulerproblem object, calls run_with_initial_data(...) and updates U.
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

  if (dealii::Utilities::MPI::this_mpi_process(app->comm_t) == 0) 
  {
    std::cout << "[INFO] Stepping on level: " 
    << level << std::endl
    << "on interval: [" << tstart << ", " << tstop << "]" << std::endl
    << "total step call number " << num_step_calls << std::endl;
  }

  //translate the fine level u coming in to the coarse level
  //this uses a function from DEALII interpolate to different mesh

  //new, coarse vector
  my_Vector u_to_step;
  reinit_to_level(&u_to_step, app, level);

  //interpolate between levels, put data from u (fine level) onto the u_to_step (coarse level)
  interpolateUBetweenLevels(u_to_step, level, *u, 0, app);
  std::cout << "past first interpolation" << std::endl;
  int count = 0;
  for(auto element: u_to_step.U)
    std::cout << "index:" << count++ << "element " << element << std::endl; 
  //step the function on this level
  app->time_loops[level]->run_with_initial_data(u_to_step.U, tstop, tstart);
  for(auto element: u_to_step.U)
    std::cout << "---------------------------------------------------/n" << "index:" << count++ << "element " << element << std::endl; 

  //interpolate this back to the fine level
  interpolateUBetweenLevels(*u,0,u_to_step,level, app);
  std::cout << "past second one" << std::endl; 
  num_step_calls++;
  //done.

  // no_nans(*app, *u, "step");//remove


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
  if (dealii::Utilities::MPI::this_mpi_process(app->comm_t) == 0)
   {
    std::cout << "[INFO] Initializing XBraid vectors" << std::endl;
   }

  //defines a coarse vector, at the coarsest level, which will be stepped, then restricted down to the fine level
  //interpolate the fine initial state into the coarse vector, then interpolates it up to the coarse level and steps
  my_Vector *u = new(my_Vector);
  my_Vector *temp_coarse = new(my_Vector);
  reinit_to_level(u, app, app->finest_index);//this is the u that we will start each time brick with.
  reinit_to_level(temp_coarse, app, app->coarsest_index);//coarse on the coarses level.
  u->U = app->levels[app->finest_index]->initial_values->interpolate(0);//sets up U data at t=0;
  
  std::cout << "in init::coarsest_index=" << app->coarsest_index << std::endl;
  std::cout << "in init::finest_index=" << app->finest_index << std::endl;
  
  //if T is not zero, we step on the coarsest level until we are done.
  //TODO: implicit assumption that T>0 always here
  if(std::abs(t) > 1e-12)
  {
    std::cout << "T>0" << std::endl;
    //interpolate the initial conditions up to the coarsest mesh
    interpolateUBetweenLevels(*temp_coarse, app->coarsest_index, *u, app->finest_index, app);
    auto str = "initialized_at_t="+std::to_string(t);
    std::cout << str << std::endl;
    //steps to the correct end time on the coarse level to end time t
    app->time_loops[app->coarsest_index]->run_with_initial_data(temp_coarse->U,t);
        print_solution(temp_coarse->U, app, t, app->coarsest_index, str);

    interpolateUBetweenLevels(*u, app->finest_index, *temp_coarse, app->coarsest_index, app);
  }
  //delete the temporary coarse U. 
  //todo: fix me!
  delete temp_coarse;

  //reassign pointer XBraid will use
  *u_ptr = u;

  std::cout << "num global vs num local " << u->U.size() << "--" << u->U.locally_owned_size() << std::endl;
  // no_nans(*app, *u, "init");//todo: remove

//
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

  // no_nans(*app, *y, "sum");//remove

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


  // no_nans(*app, *u, "spatialnorm");//removed

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
  UNUSED(app);
  UNUSED(u);

  //state what iteration we are on
  braid_AccessStatusGetIter(astatus, &mgCycle);
  braid_AccessStatusGetT(astatus, &t);

  if (dealii::Utilities::MPI::this_mpi_process(app->comm_t) == 0)
  {
    std::cout << "Cycles done: " << mgCycle << std::endl;
  }

  //calculate drag and lift of this u and output to terminal//TODO: better output this
  //calculateDragAndLift(u, app);
  std::string fname = "./cycle" + std::to_string(mgCycle);
  print_solution(u->U, app, t, 0/*level, always needs to be zero, to be fixed*/, fname);

  //calculate drag and lift for this solution on level 0


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
{//TODO: test this with n_dof output
  if (dealii::Utilities::MPI::this_mpi_process(app->comm_t) == 0)
  {
    std::cout << "[INFO] BufPack Called" << std::endl;
  }

  const int problem_dimension = app->problem_dimension;
  NUMBER *dbuffer = (NUMBER*)buffer;
  unsigned int n_dof = app->levels[0]->offline_data->dof_handler().n_dofs();//number of dofs at finest level
  unsigned int buf_size = n_dof * app->problem_dimension;
  dbuffer[0] = buf_size + 1;//buffer + size
  dealii::Tensor<1, problem_dimension, NUMBER> temp_tensor;
  // dealii::Tensor<1,problem_dimension, dealii::VectorizedArray<NUMBER>> temp_tensor_vectorized;
  for(unsigned int node=0; node < n_dof; node++)
  {
    //extract tensor at this node
    temp_tensor = u->U.get_tensor(node);//TODO:FIXME: test linear method for speed...
    // std::cout << "after get_tensor" << std::endl;
    for (unsigned int component = 0; component < problem_dimension; ++component) {
      Assert(buf_size >= (node + component),
             ExcMessage("In my_BufPack, the size of node + component is "
                        "greater than the buff_size (the expected size of the vector)."));
      dbuffer[problem_dimension*(node) + component + 1] = u->U[problem_dimension*node + component];
    }
  }
  // std::cout << "after loop" << std::endl;
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

  // the vector should be size (dim + 2) X n_dofs at finest level.
  my_Vector *u = NULL; // the vector we will pack the info into
  u = new(my_Vector);//TODO: where does this get deleted? Probably wherever owns the u_ptr.
  reinit_to_level(u, app, app->finest_index);//each U is at the finest level.
  dealii::Tensor<1, problem_dimension, NUMBER> temp_tensor;

  //unpack the sent data into the right level
  for(unsigned int node=0; node < app->n_fine_dofs; node++)
  {
    //get tensor at node.
    for(unsigned int component = 0; component < app->problem_dimension; ++component)
    {
      // temp_tensor[component] = dbuffer[app->problem_dimension*node + component + 1];//+1 because buffer_size = n_dof + 1
      u->U[app->problem_dimension*node + component] = dbuffer[app->problem_dimension*node + component + 1];//test for speed.
      Assert(node + component +1 <= buf_size,
          ExcMessage("somehow, you are exceeding the buffer size as you unpack"));
    }
    //insert tensor at node
    // u->U.write_tensor(temp_tensor, node);//TODO:FIXME: try linear method?
  }

  *u_ptr = u;//modify the u_ptr does this create a memory leak as we just point this pointer somewhere else?

  // no_nans(*app, *u, "unpack");

//  std::cout << "Buffunpack done." << std::endl;
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
  MPI_Comm comm_world = MPI_COMM_WORLD;//create MPI_object
  //scoped MPI object, no need to call finalize at the end.
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);  //create objects

  //split the object into the number of time processors, and the number of spatial processors per time chunk.
  MPI_Comm comm_x, comm_t;
  const int px = 1; //one spatial processors to use per time brick
  std::cout << "Number of Proc specified: " << px << std::endl;

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

  //set up app and all underlying data, initialize parameters
  my_App app(comm_x, comm_t);
  dealii::ParameterAcceptor::initialize("test.prm");
  app.prepare();//call after initialize the parameters to the test.prm file

  //initialize the time_averaged vector to the finest level
  // my_Vector time_averaged;
  // reinit_to_level(&time_averaged, app, 0/*the finest level*/);

  /* Initialize Braid */
  braid_Core core;
  double tstart = 0.0;
  double tstop = 5.0;
  int ntime = 4;//this should in general be the number of time bricks you want. You need to ensure that px * ntime = TOTAL NUMBER PROCESSORS

  braid_Init(comm_world,
             comm_t,
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
  auto max_level_index =
      std::max_element(app.refinement_levels.begin(), app.refinement_levels.end());
  unsigned int max_levels = app.refinement_levels.size(); // fixme, cthis is later cast to an int.
  int nrelax = 1; // FIXME: add this and delete the UNUSED(nrelax) later.
  //      int       skip          = 0;
  double tol = 1e-2;
  // int       cfactor       = 2;
  int max_iter = 0;
  //      int       min_coarse    = 10;
  // int       fmg           = test_case.fmg;
  // int       scoarsen      = 0;
  // int       res           = 0;
  // int       wrapper_tests = 0;
  int print_level = 2;
  /*access_level=1 only calls my_access at end of simulation*/
  int access_level = 2;
  int use_sequential = 0;

  UNUSED(nrelax);

  braid_SetPrintLevel(core, print_level);
  braid_SetAccessLevel(core, access_level);
  braid_SetMaxLevels(core, max_levels);
  //             braid_SetMinCoarse( core, min_coarse );
  //             braid_SetSkip(core, skip);
  //      braid_SetNRelax(core, -1, nrelax);
  braid_SetAbsTol(core, tol);
  //       braid_SetCFactor(core, -1, cfactor);
  braid_SetMaxIter(core, max_iter);
  braid_SetSeqSoln(core, use_sequential);

  std::cout << "before braid_drive\n";
  braid_Drive(core);


  // Free the memory now that we are done
  braid_Destroy(core);


}
