#include <vector>
#include <string>
#include <memory>

//ryujin includes
#include "euler/description.h"

//MPI
#include <deal.II/base/mpi.h>

//deal.II includes
#include <deal.II/base/parameter_acceptor.h>

//xbraid include
#include <braid.h>
#include <braid.hpp>

//mgrit includes
#include "level_structures.h"

/*
 * Author: Jerett Cherry, Colorado State University
 */

/**
 *   \brief Contains the implementation of the mandatory X-Braid app and vector.
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

// This struct contains all data that changes with time. For now
// this is just the solution data at interpolated always to the finest level.
/**
 * \brief Struct that contains a ryujin::TimeLoop::vector_type of size problem_dimension*n_dofs.
 */

class MyVector
{
public:
  // Vector type
  ryujin::TimeLoop<ryujin::Euler::Description, 2, NUMBER>::vector_type U;

  // Constructor
  MyVector();

  // Destructor
  ~MyVector();
};

/**
 * \brief Cpp wrapper for all of the required functions and data.
*/
class MyApp : public BraidApp, public dealii::ParameterAcceptor
{

  using Description = ryujin::Euler::Description;
  using Number = NUMBER;
  using LevelType
       = std::shared_ptr<ryujin::mgrit::LevelStructures<Description, 2, Number>>;
  using TimeLoopType
       = std::shared_ptr<ryujin::TimeLoop<Description,2,Number>>;
  using HyperbolicSystemView =
      typename Description::HyperbolicSystem::template View<2, Number>;
  static constexpr unsigned int problem_dimension =
      HyperbolicSystemView::problem_dimension;
  static constexpr unsigned int n_precomputed_values =
      HyperbolicSystemView::n_precomputed_values;
  using scalar_type = ryujin::OfflineData<2, Number>::scalar_type;
  using vector_type = ryujin::MultiComponentVector<Number, problem_dimension>;//TODO: determine if I need these typenames at all in app;
  using precomputed_type = ryujin::MultiComponentVector<Number, n_precomputed_values>;

protected:
  //BraidApp defines comm_t_, t_start_, tstop_, ntime_ the rest need to be defined below.

public:
  //Constructor takes communicators and a vector representing how much refinement is needed at each level. 
  //Also defines the number of levels.
  MyApp(const MPI_Comm comm_x = MPI_COMM_WORLD,
        const MPI_Comm comm_t = MPI_COMM_WORLD,
        const std::vector<unsigned int> a_refinement_levels);

  /** @brief Apply the time stepping routine to the input vector @a u
       corresponding to time @a tstart, and return in the same vector @a u the
       computed result for time @a tstop. The values of @a tstart and @a tstop
       can be obtained from @a pstatus.

       @param[in,out] u Input: approximate solution at time @a tstart.
                         Output: computed solution at time @a tstop.
       @param[in] ustop Previous approximate solution at @a tstop?
       @param[in] fstop Additional source at time @a tstop. May be set to NULL,
                         indicating no additional source.
  */

  /// Initialize with prm file.
  void initialize(const std::string prm_file);
  // TODO: write access functions for things like levels[level],
  // timeloops[level]

private:
  /// Creates all objects ryujin needs to run.
  void create_mg_levels();
  /// Calls prepare on all objects ryujin needs to run.
  void prepare_mg_objects();

public: // Braid Required Routines

  virtual braid_Int Step(braid_Vector u,
                         braid_Vector ustop,
                         braid_Vector fstop,
                         BraidStepStatus &pstatus);

  /** @brief Compute the residual at time @a tstop, given the approximate
      solutions at @a tstart and @a tstop. The values of @a tstart and @a tstop
      can be obtained from @a pstatus.

      @param[in]     u Input: approximate solution at time @a tstop.
      @param[in,out] r Input: approximate solution at time @a tstart.
                        Output: residual at time @a tstop.
  */
  virtual braid_Int
  Residual(braid_Vector u, braid_Vector r, BraidStepStatus &pstatus);

  /// Allocate a new vector in @a *v_ptr, which is a deep copy of @a u.
  virtual braid_Int Clone(braid_Vector u, braid_Vector *v_ptr);

  /** @brief Allocate a new vector in @a *u_ptr and initialize it with an
      initial guess appropriate for time @a t. If @a t is the starting time,
      this method should set @a *u_ptr to the initial value vector of the ODE
      problem. */
  virtual braid_Int Init(braid_Real t, braid_Vector *u_ptr);


  /// De-allocate the vector @a u.
  virtual braid_Int Free(braid_Vector u);

  /// Perform the operation: @a y = @a alpha * @a x + @a beta * @a y.
  virtual braid_Int
  Sum(braid_Real alpha, braid_Vector x, braid_Real beta, braid_Vector y);

  /// Compute in @a *norm_ptr an appropriate spatial norm of @a u.
  virtual braid_Int SpatialNorm(braid_Vector u, braid_Real *norm_ptr);

  /// @see braid_PtFcnAccess.
  virtual braid_Int Access(braid_Vector u, BraidAccessStatus &astatus);

  /// @see braid_PtFcnBufSize.
  virtual braid_Int BufSize(braid_Int *size_ptr, BraidBufferStatus &bstatus);

  /// @see braid_PtFcnBufPack.
  virtual braid_Int
  BufPack(braid_Vector u, void *buffer, BraidBufferStatus &bstatus);

  /// @see braid_PtFcnBufUnpack.
  virtual braid_Int
  BufUnpack(void *buffer, braid_Vector *u_ptr, BraidBufferStatus &bstatus);

public: // Public Data

  const MPI_Comm comm_x; //todo: can I make this const? uninitialized communicators now.
  std::vector<LevelType> levels; //instantiation
  std::vector<unsigned int> refinement_levels;
  std::vector<TimeLoopType> time_loops;
  braid_Int finest_index, coarsest_index;
  unsigned int n_fine_dofs;
  unsigned int n_locally_owned_dofs;
  bool print_solution = false;
  braid_Int num_time;
  braid_Int cfactor;//FIXME: decide what types here? ryujin types or braid types? what about doubles??
  braid_Int max_iter;
  bool use_fmg;
  braid_Int n_relax;
  unsigned int n_cycles = 0;
  braid_Int access_level;
};