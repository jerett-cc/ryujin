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

/*Includes*/

/*-------- Third Party --------*/

/*-------- Project --------*/
//#include "offlinedata.h"
//#include "parallel-in-time-euler.h"

// This preprocessor macro is used on function arguments
// that are not used in the function. It is used to
// suppress compiler warnings.
#define UNUSED(x) (void)(x)


/**
 * This function should take a vector and resize it with
 * the size specified TODO: delete this, do not need it anymore.
 */

//template<typename Description, typename V, int dim>
//void resizeVector(V& v,
//                  const int size,
//                  const int num_comps = ryujin::TimeLoop<Description,
//                                                         dim,
//                                                         NUMBER>::problem_dimension)
//{
//  for(int i = 0; i < num_comps; ++i)
//    v[i].reinit(size);
//}


// This struct contains all data that changes with time. For now
// this is just the solution data. When doing AMR this should
// probably include the triangulization, the sparsity patter,
// constraints, etc.
/**
 * \brief Struct that contains the deal.ii vector of size (dim+2,n_dofs).
 */
typedef struct _braid_Vector_struct
{
  ryujin::TimeLoop<ryujin::Euler::Description, 2, NUMBER>::vector_type U;//change this to
} my_Vector;


// This struct contains all the data that is unchanging with time.
/**
 *
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
    using scalar_type = typename ryujin::OfflineData<2, Number>::scalar_type;
    using vector_type = ryujin::MultiComponentVector<Number, problem_dimension>;//TODO: determine if I need these typenames at all in app;
    using precomputed_type = ryujin::MultiComponentVector<Number, n_precomputed_values>;


    const MPI_Comm comm_x, comm_t;
    std::vector<LevelType> levels; //instantiation
    std::vector<unsigned int> refinement_levels;

    std::vector<TimeLoopType> time_loops;
    int finest_index, coarsest_index;

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
      levels[0] = std::make_shared<ryujin::mgrit::LevelStructures<Description, 2, Number>>(comm_x, 0);
      time_loops[0] = std::make_shared<ryujin::mgrit::TimeLoopMgrit<Description, 2, Number>>(comm_x, *levels[0],0,0);

      //default objects. their member variables will be modified when
      //parameteracceptor initialize is called, then you can call prepare.
      //FIXME: a potential bug can be introduced when a user calls prepare()
      //before they call parameterhandler::initialize, how to stop this?
    };

    //This function NEEDS to be called after parameter handler has been initialized
    //otherwise user defined refinement levels are not going to be used, only the
    //default above.
    void prepare()
    {
      //reset the coarsest index
      coarsest_index = refinement_levels.size()-1;

      //resize the vectors
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
        levels[i] = std::make_shared<ryujin::mgrit::LevelStructures<Description, 2, Number>>(comm_x, refinement_levels[i]);
        time_loops[i] = std::make_shared<ryujin::mgrit::TimeLoopMgrit<Description,2,Number>>(comm_x, *(levels[i]), 0,0);
      }
    }
} my_App;

/**
 * @brief This function is used to interpolate a vector (from_V) on a certain mesh
 *        to a nother interpolated solution on another mesh.
 *
 * @param[out] to_v, the vector V to which we are interpolating the solution
 * @param[in]  the mg level we are interpolating to
 * @param[in]  from_V the vector storing the solution we wish to interpolate
 * @param[in]  from_level the mg level which currently stores the solution
 * @param[in]  app, the user defined app which stores needed information about the mg levels
 *
 * NOTE: this function only works if the vectors to_v and from_v have an at() method implemented.
 */
template<typename V>
void interpolateUBetweenLevels(V& to_v,
                               const int to_level,
                               const V& from_v,
                               const int from_level,
                               braid_App& app)
{
  UNUSED(to_v);
  UNUSED(to_level);
  UNUSED(from_v);
  UNUSED(from_level);
  UNUSED(app);

//  assert(to_v.at(0).size() == app->TI_p[to_level]->offline_data.dof_handler.n_dofs()
//      && "trying to interpolate to a vactor and level where the n_dofs do not match");
//
//  unsigned int n_solution_variables = ProblemDescription<dim>::n_solution_variables;//constant at any dimension
//  for(unsigned int comp=0; comp<n_solution_variables; comp++)
//    {
//      VectorTools::interpolate_to_different_mesh(app->TI_p[from_level]->offline_data.dof_handler,
//                                                 from_v.at(comp),
//                                                 app->TI_p[to_level]->offline_data.dof_handler,
//                                                 to_v.at(comp));
//    }
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
            braid_Vector     u,
            braid_StepStatus status)
{
  //this variable is used for writing data to
  //different files during the parallel computations.
  //is passed to run_with_initial_data
  static unsigned int num_step_calls = 0;
  std::cout << "step called\n";

  //grab the MG level for this step
  int level;
  braid_StepStatusGetLevel(status, &level);
  std::cout << "with level= " << level << std::endl;

  //use a macro to get rid of some unused variables to avoid -Wall messages
  UNUSED(ustop);
  UNUSED(fstop);
  UNUSED(app);
  UNUSED(u);
  //grab the start time and end time
  double tstart;
  double tstop;
  braid_StepStatusGetTstartTstop(status, &tstart, &tstop);

  //translate the fine level u coming in to the coarse level
  //this uses a function from DEALII interpolate to different mesh
  my_Vector u_to_step;
  std::cout << "size of U_to_step: " << u_to_step.U.size() << std::endl;
  const auto coarse_offline_data = app->levels.at(level)->offline_data;
  const auto num_coarse_dof = coarse_offline_data->dof_handler().n_dofs();
  const auto coarse_size = num_coarse_dof*app->problem_dimension;
  u_to_step.U.reinit_with_scalar_partitioner(coarse_offline_data->scalar_partitioner());
  std::cout << "size of U_to_step after reinit: " << u_to_step.U.size() << std::endl;

  exit(1);
//
//  //interpolate the data coming in with u (finest level) onto the
//  //u_to_step (coarse level)
//  interpolateUBetweenLevels(u_to_step,
//                            level,
//                            u->data,
//                            app->finest_index,
//                            app);
//
//
//  //set up the object that is going to do the time stepping
//  //the vector size here should match that of the right level.
//  EulerEquation<dim> eq(app->TI_p[level], tstart, tstop);
//
//  //set the level
//  eq.setLevel(level);
//
//  //set the mg_cycle
//  eq.setCycle(app->mg_cycle);
//
//  std::cout << "_____________Level: " << level << "\n"
//            << "ndof at this level: " << eq.get_size() << std::endl;
//  //update u_interpolated with a step
//  u_to_step = eq.run_with_initial_data(u_to_step,
//                                       tstart,
//                                       tstop,
//                                       num_step_calls);
//  ++num_step_calls;
//
//  //now that we have completed the step, we interpolate back to the
//  //finest level from the current level and coarse solution
//  interpolateUBetweenLevels(u->data,
//                            app->finest_index,
//                            u_to_step,
//                            level,
//                            app);
//  //now u is updated with the stepped solution.
////  std::cout << "Number of my step calls: " << num_step_calls << "\n";
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
  std::cout << "Init called.\n";
//
//  //static unsigned int initcall = 0;
  my_Vector *u = new(my_Vector);
  UNUSED(app);
  UNUSED(t);
  UNUSED(u_ptr);

  //initializes all at a fine level
//  resizeVector(u->data, app->vect_size);
//
//  //if t is not zero, we integrate the initial solution from 0->t on the coarsest level, and
//  //get a coarse solution.
//  if(std::abs(t-0) > 1e-10)//NOTE: FIXME?: this could in theory cause problems when the time bricks are so many that one has an end time before 10^-10.
//  {
//    //create the time stepping object
//    EulerEquation<dim> eq(app->TI_p[app->coarsest_index], 0, t);
//    std::unique_ptr<my_Vector> u_coarse = std::make_unique<my_Vector>();//coarse vector
//    resizeVector(u_coarse->data,
//                 app->TI_p[app->coarsest_index]->offline_data.dof_handler.n_dofs());
//
//    //interpolate the fine mesh to the appropriate mesh
//    interpolateUBetweenLevels(u_coarse->data, app->coarsest_index,
//                              u->data, app->finest_index,
//                              app);
//    //run with initial data
//    std::cout << "end time " << t << std::endl;
//    eq.run_with_initial_data(u_coarse->data, 0, t, 0, false);
//
//    //interpolate the coarse mesh back to the fine mesh.
//    interpolateUBetweenLevels(u->data, app->finest_index,
//                              u_coarse->data,app->coarsest_index,
//                              app);
//  }
////  //else do nothing to U, as we will initialize time = 0 inside the run function.
////  EulerEquation<dim> eq(app->TI_p[app->finest_index], 0, t);
////  unsigned int procID = Utilities::MPI::this_mpi_process(app->comm_t);
////  eq.call = initcall++;
////  eq.output(u->data, "init_test", t, procID);
//
//  *u_ptr = u;
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
  UNUSED(app);
  UNUSED(u);
  UNUSED(v_ptr);

//  std::cout << "Clone called" << std::endl;
//  UNUSED(app);
//  my_Vector *v = new(my_Vector);
//  int size = u->data[0].size();
////  Assert(size == app->TI.offline_data.dof_handler.n_dofs(),
////      "size of cloned vector does not match the number of dof's. It should.");
//
//  for(unsigned int i=0;
//        i < ProblemDescription<dim>::n_solution_variables;
//        ++i)
//    {
//      v->data.at(i).reinit(size);
//    }
//
//  for(unsigned int i=0;
//          i < ProblemDescription<dim>::n_solution_variables;
//          ++i)
//      {
//        v->data.at(i) = u->data.at(i);
//      }
//    *v_ptr = v;
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
  std::cout << "Free called" << std::endl;
  UNUSED(app);
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
  UNUSED(alpha);
  UNUSED(x);
  UNUSED(beta);
  UNUSED(y);


//  std::cout << "Sum called\n";
//  UNUSED(app);
//  LinearAlgebra::distributed::Vector<double> tmp1, tmp2;
//  int x_size = x->data[0].size();//size of x
//  int y_size = y->data[0].size();
//  assert(x_size == y_size && "adding vectors of different sizes will not work");
//
//  tmp1.reinit(x_size);
//  tmp2.reinit(y_size);
//  for(unsigned int i=0;
//      i < ProblemDescription<dim>::n_solution_variables;
//      ++i)
//  {
//    tmp1 = x->data.at(i);
//    tmp2 = y->data.at(i);
//    tmp1*=alpha;
//    tmp2*=beta;
//    tmp1+=tmp2;
//
//    y->data.at(i) = tmp1;
//  }
//
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
  UNUSED(u);
  UNUSED(norm_ptr);

//  std::cout << "Norm called" << std::endl;
//  UNUSED(app);
//  double l2 = 0;
//  for(unsigned int i=0;
//        i < ProblemDescription<dim>::n_solution_variables;
//        ++i)
//    {
//      l2 +=u->data.at(i).norm_sqr();
//    }
//  *norm_ptr = std::sqrt(l2);
//
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
  UNUSED(app);
  UNUSED(u);
  UNUSED(astatus);

//  static int mgCycle = 0;
//  std::cout << "Access called" << std::endl;
//  //get the iteration
//  braid_AccessStatusGetIter(astatus, &mgCycle);
//  std::cout << "Cycles done: " << mgCycle << std::endl;
//  app->mg_cycle = mgCycle;
//  UNUSED(u);
//  UNUSED(astatus);
//
//  //for now, we just increment the mg_cycle
//
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
  UNUSED(app);
  UNUSED(size_ptr);
  UNUSED(bstatus);

//  UNUSED(bstatus);
//  int size = app->vect_size * app->n_solution_variables;//no vector can be bigger than this, so we are very conservative.
//  *size_ptr = (size+1) * sizeof(double);//all values are doubles +1 is for the size of the buffers
  return 0;
}

/**
 *  \brief Linearizes a vector to be sent to another processor
 *
 *  Linearizes (packs) a data buffer with the contents of
 *  some solution state u.
 *
 *  \param app - The braid app struct containing user data
 *  \param u The vector that must be packed into buffer
 *  \param buffer The buffer that must be filled with u
 *  \param bstatus The XBraid status structure
 *  \return Success (0) or failure (1)
 */
int
my_BufPack(braid_App           app,
           braid_Vector        u,
           void               *buffer,
           braid_BufferStatus  bstatus)
{
  UNUSED(app);
  UNUSED(u);
  UNUSED(buffer);
  UNUSED(bstatus);


//  std::cout << "Buff pack called" << std::endl;
//  UNUSED(app);
//  double *dbuffer = (double*)buffer;
//
//  unsigned int n_dof = app->vect_size;//number of dofs at fines level
//  assert(n_dof == u->data.at(0).size() && "the number of degrees of freedom do not match");
//  unsigned int n_solution_variables = app->TI_p[0]->time_stepping.n_solution_variables;//total problem dimension.(spacedim + density + energy)
//  unsigned int buf_size = n_dof * n_solution_variables;
//  dbuffer[0] = buf_size + 1;//buffer + size
//
//  for(unsigned int j=0; j < n_solution_variables; ++j)
//  {
//    for(unsigned int i=0; i != n_dof; ++i)
//    {
//      dbuffer[j*n_dof + i + 1] = (u->data).at(j)(i);
//    }
//  }
//
//  braid_BufferStatusSetSize(bstatus, (buf_size+1)*sizeof(double));
//
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
  UNUSED(app);
  UNUSED(buffer);
  UNUSED(u_ptr);
  UNUSED(bstatus);

//  // the vector should be size (dim + 2) X n_dofs at finest level.
//  std::cout << "buff unpack called\n";
//  UNUSED(bstatus);
//  my_Vector *u = NULL; // the vector we will pack the info into
//  double *dbuffer = (double*)buffer;
//  int buf_size = static_cast<int>(dbuffer[0]);
//
//  u = new(my_Vector);//where does this get deleted?
//
//  resizeVector(u->data, app->vect_size, app->n_solution_variables);
//
//  //unpack the sent data into the right level
//  for(int j=0; j < app->n_solution_variables; ++j)
//  {
//    for(int i = 0; i < app->vect_size; ++i)
//    {
//      u->data.at(j)(i) = dbuffer[j*app->vect_size + i + 1];//+1 because buffer_size = n_dof + 1
//      assert(j*app->vect_size + i +1 <= buf_size && "somehow, you are exceeding the buffer size as you unpack");
//    }
//  }
//
//  *u_ptr = u;//modify the u_ptr does this create a memory leak as we just point this pointer somewhere else?
//
//  std::cout << "Buffunpack done." << std::endl;
  return 0;
}

int main(int argc, char *argv[])
{
  const MPI_Comm comm = MPI_COMM_WORLD;
  //scoped MPI object, no need to call finalize at the end.
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);  //create objects

  //todo: change this to a call to something similar to the main ryujin executable. problem_dispach??

  //set up app and all underlying data, initialize parameters
  my_App app(comm, comm);
  dealii::ParameterAcceptor::initialize("test.prm");

  //initialize time loop and call parameter acceptor initialize again
//  ryujin::mgrit::TimeLoopMgrit<ryujin::Euler::Description,2,double> time_loop(app.comm_x, *(app.levels.at(0)),0,0.1);
//  dealii::ParameterAcceptor::initialize("test.prm");


//  std::cout << app.levels.size() << std::endl;
//  std::cout << app.refinement_levels.size() << std::endl;

  app.prepare();//call after initialize the parameters
  std::cout << app.levels.size() << std::endl;//check parameters again
  std::cout << app.refinement_levels.size() << std::endl;


//  time_loop.run();
//  std::cout << "Size of U: " << time_loop.get_U().size() << std::endl;

  for(const auto xi : app.refinement_levels)
    std::cout << xi << std::endl;
//  std::cout << app.levels.at(0)->offline_data->dof_handler().n_dofs() << std::endl;
//  ryujin::TimeLoop<ryujin::Euler::Description, 2, NUMBER> timeloop(comm);
//  dealii::ParameterAcceptor::initialize("cylinder-parameters.prm");//initialize the file specified by the user


}
