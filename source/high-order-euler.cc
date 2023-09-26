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

//MPI
#include <deal.II/base/mpi.h>

//ryujin includes
//#include "hyperbolic_module.h"
//#include "offline_data.h"
//#include "geometry_cylinder.h"
//#include "discretization.h"
#include "hyperbolic_system.h"
#include "time_loop.h"
#include "euler/description.h"
#include "convenience_macros.h"

//deal.II includes
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/mpi.h>

//xbraid include
#include <braid.h>
#include <braid_test.h>

/*
 * Author: Jerett Cherry, Colorado State University
 */

const int dim = 2;
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
#include <deal.II/numerics/vector_tools.h>

/*-------- Project --------*/
//#include "offlinedata.h"
//#include "parallel-in-time-euler.h"

// This preprocessor macro is used on function arguments
// that are not used in the function. It is used to
// suppress compiler warnings.
#define UNUSED(x) (void)(x)

/**
 * This function should take a vector and resize it with
 * the size specified
 */

template<typename Description, typename V, int dim>
void resizeVector(V& v,
                  const int size,
                  const int num_comps = ryujin::TimeLoop<Description,
                                                         dim,
                                                         NUMBER>::problem_dimension)
{
  for(int i = 0; i < num_comps; ++i)
    v[i].reinit(size);
}


// This struct contains all data that changes with time. For now
// this is just the solution data. When doing AMR this should
// probably include the triangulization, the sparsity patter,
// constraints, etc.
/**
 * \brief Struct that contains the deal.ii vector of size (dim+2,n_dofs).
 */
typedef struct _braid_Vector_struct
{
  ryujin::TimeLoop<ryujin::Euler::Description, dim, NUMBER>::vector_type U;

} my_Vector;

// This struct contains all the data that is unchanging with time.
/**
 * \brief Struct that contains the HeatEquation and final
 * time step number.
 *
 * XBRAID uses this app structure as information that each
 * process can use when computing, so naturally should be
 * used to store global information relevant to each
 * process.
 *
 * The app stores spatial and temporal communicators,
 * comm_x and comm_t. (For this code, we use MPI_COMM_SELF for
 * comm_x, and MPI_COMM_WORLD for comm_t. For maximum performance,
 * one needs to leverage the spatial parallelism that is build in
 * to step-69, and coordinate with the additional parallelism grant-
 * ed by XBRAID. To do this, one should use a custom communicator
 * for comm_x, but MPI_COMM_WORLD is a natural choice for comm_t
 * since it oversees the communication between 'time bricks'.
 *
 * For step-69 to run parallel in time, one needs only to store
 * objects like the sparsity pattern, dof-handler, etc once.
 * Then, this information can be queried by the individual
 * processors doing the work. This information is stored in
 * the TimeIndependent class--reference its documentation.
 *
 * XBRAID
 */

typedef struct _braid_App_struct
{
//  public:
//    MPI_Comm comm_x, comm_t;
////    TimeIndependent<dim> TI;
//    std::vector<std::shared_ptr<TimeIndependent<dim>>> TI_p;
//    int final_step, mg_cycle;
//    int mg_level;//FIXME remove this
//    std::vector<unsigned int> refinement_levels;
//    int vect_size, finest_index, coarsest_index, n_solution_variables;
//    _braid_App_struct(MPI_Comm mpi_comm_x,
//                      const MPI_Comm comm_t,
//                      int final_step,
//                      const unsigned int mg_level,
//                      const std::vector<unsigned int> refine_levels)
//    : comm_x(mpi_comm_x),
//      comm_t(comm_t),
//      TI_p(refine_levels.size()),
//      final_step(final_step),
//      mg_cycle(0),//cycle starts at 0 since there has been no cycles yet
//      mg_level(mg_level),
//      refinement_levels(refine_levels)
//    {
//      //assert that user specified levels vector is ordered properly
//      for(unsigned int i = 0; i < refinement_levels.size()-1; ++i)
//      {
//        assert(refinement_levels.at(i) <= refinement_levels.at(i+1)
//            && "in testcase.prm the levels need to be in increasing order");
//      }
//
//      finest_index = 0;
//      int last_index = refinement_levels.size()-1;
//      coarsest_index = last_index;
//      for (unsigned int i=0; i<refinement_levels.size(); i++)
//      {
//        std::cout << "_____________________________\n";
//        std::cout << "Refinement i=" << i << "\n";
//        /*set up the array of time independent objects.*/
//
//        //the finest level should be at level 0 (i=0), ensuring that level queries
//        //later on match the xbraid standard
//        TI_p.at(i) = std::make_shared<TimeIndependent<dim>>(mpi_comm_x,
//                                                            comm_t,
//                                                            refinement_levels.at(last_index - i));
//        /*new*/
//        TI_p[i]->pcout << "Reading parameters and allocating objects... " << std::flush;
//        TI_p[i]->pcout << "done" << std::endl;
//        {
//          print_head(TI_p[i]->pcout, "create triangulation");
//
//          TI_p[i]->discretization.setup();
//
//          if (TI_p[i]->resume)
//            TI_p[i]->discretization.triangulation.load(TI_p[i]->base_name + "-checkpoint.mesh");
//          else
//            TI_p[i]->discretization.triangulation.refine_global(TI_p[i]->discretization.refinement);
//
//          TI_p[i]->pcout << "Number of active cells:       "
//              << TI_p[i]->discretization.triangulation.n_global_active_cells()
//              << std::endl;
//
//          //FIXME this piece should be the same for every process, find a way to break it out.
//          print_head(TI_p[i]->pcout, "compute offline data");
//          TI_p[i]->offline_data.setup();
//          TI_p[i]->offline_data.assemble();//FIXME to here, we should not need anything past this??
//
//          TI_p[i]->pcout << "Number of degrees of freedom: "
//              << TI_p[i]->offline_data.dof_handler.n_dofs() << std::endl;
//
//          print_head(TI_p[i]->pcout, "set up time step");
//          TI_p[i]->time_stepping.prepare();
//          TI_p[i]->schlieren_postprocessor.prepare();
//        }
//      }//loop
//      vect_size =  TI_p.at(finest_index)->offline_data.dof_handler.n_dofs();
//      n_solution_variables = ProblemDescription<dim>::n_solution_variables;
//      ParameterAcceptor::initialize("step-69.prm");
//
//      std::cout << "APP created.\n";
//      /*new end*/
//    };
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
//  //this variable is used for writing data to
//  //different files during the parallel computations.
//  //is passed to run_with_initial_data
//  static unsigned int num_step_calls = 0;
//  std::cout << "step called\n";
//
//  //grab the MG level for this step
//  int level;
//  braid_StepStatusGetLevel(status, &level);
//  std::cout << "with level= " << level << std::endl;
//
//  //use a macro to get rid of some unused variables to avoid -Wall messages
//  UNUSED(ustop);
//  UNUSED(fstop);
//  //grab the start time and end time
//  double tstart;
//  double tstop;
//  braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
//
//  //translate the fine level u to the coarse level
//  //this uses a function from DEALII interpolate to different mesh
//  EulerEquation<dim>::vector_type u_to_step;
//  int coarse_size = app->TI_p.at(level)->offline_data.dof_handler.n_dofs();
//  resizeVector(u_to_step,coarse_size);
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
//  return 0;
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
//  std::cout << "Init called.\n";
//
//  //static unsigned int initcall = 0;
//  my_Vector *u = new(my_Vector);
//  //initializes all at a fine level
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
//  return 0;
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
//    return 0;
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
//  return 0;
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
//  return 0;
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
//  return 0;
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
//  UNUSED(bstatus);
//  int size = app->vect_size * app->n_solution_variables;//no vector can be bigger than this, so we are very conservative.
//  *size_ptr = (size+1) * sizeof(double);//all values are doubles +1 is for the size of the buffers
//  return 0;
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
//  return 0;
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
//  return 0;
}

int main(int argc, char *argv[])
{
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Init(&argc, &argv);
  //create objects

  //todo: change this to a call to something similar to the main ryujin executable. problem_dispach??

  ryujin::TimeLoop<ryujin::Euler::Description, 2, NUMBER> timeloop(comm);
  dealii::ParameterAcceptor::initialize("cylinder-parameters.prm");//initialize the file specified by the user

  timeloop.run();

  MPI_Finalize();


}
