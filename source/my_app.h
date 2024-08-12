#pragma once
#include <vector>
#include <string>
#include <memory>

//ryujin includes
#include <compile_time_options.h>
#include "time_loop.h"
#include "scope.h"


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
namespace mgrit{

  template<typename Number, typename Description, int dim>
  class MyVector
  {
    using HyperbolicSystemView =
        typename Description::template HyperbolicSystemView<dim, Number>;
    static constexpr unsigned int problem_dim =
        HyperbolicSystemView::problem_dimension;
    static constexpr unsigned int n_prec =
        HyperbolicSystemView::n_precomputed_values;

  public:
    // Vector type
    ryujin::Vectors::StateVector<Number, problem_dim, n_prec> U;

    // Constructor
    MyVector() {};

    // Destructor
    ~MyVector() {};
  };

  /**
   * \brief Cpp wrapper for all of the required functions and data.
   */
  template<typename Number, typename Description, int dim>
  class MyApp : public BraidApp, public dealii::ParameterAcceptor
  {
    public:
    using LevelType =
        std::shared_ptr<ryujin::mgrit::LevelStructures<Description, dim, Number>>;
    using TimeLoopType =
        std::shared_ptr<ryujin::TimeLoop<Description, dim, Number>>;
    using OfflineDataType = ryujin::OfflineData<dim,Number>;
    using DiscretizationType = ryujin::Discretization<dim>;
    using HyperbolicSystemView =
        typename Description::template HyperbolicSystemView<dim, Number>;
    static constexpr unsigned int problem_dimension =
        HyperbolicSystemView::problem_dimension;
    static constexpr unsigned int n_precomputed_values =
        HyperbolicSystemView::n_precomputed_values;
    using scalar_type = ryujin::Vectors::ScalarVector<Number>;
    using vector_type =
        ryujin::Vectors::MultiComponentVector<Number,
                                     problem_dimension>; // TODO: determine if I
                                                         // need these typenames
                                                         // at all in app;
    using StateVector = ryujin::Vectors::StateVector<Number, problem_dimension, n_precomputed_values>;
    using precomputed_type =
        ryujin::Vectors::MultiComponentVector<Number, n_precomputed_values>;
    
    using my_vector = MyVector<Number, Description, dim>;

    /// @brief Constructor.
    /// @param comm_x Spatial communicator to be used by ryujin.
    /// @param comm_t Temporal Communicator to be used by braid.
    /// @param a_refinement_levels A vector of number of global refinements. eg.
    /// {1 2 4} defines three temporal grids where the spatial mesh has been
    /// refined 4 times on the finest, 2 times on the middle, and once at
    MyApp(const MPI_Comm comm_x,
          const MPI_Comm comm_t,
          const std::vector<int> a_refinement_levels);

    /// @brief Destructor.
    virtual ~MyApp();

    /// @brief Initializes all data with the parameters from the specified file.
    /// @param prm_file
    void initialize(const std::string prm_file);
    // TODO: write access functions for things like levels[level],
    // timeloops[level]

    /// @brief Reinitializes a vector to the level we wish. This changes the
    /// datastructures that are associated to @ u.
    /// @param u
    /// @param level
    void reinit_to_level(my_vector *u, const int level);

    /// @brief Interpolates a vector to a vector which may live on a different
    /// spatial mesh.
    /// @param to_v The vector to which we are interpolating.
    /// @param to_level The level this vector lives at.
    /// @param from_v The vector we wish to interpolate from.
    /// @param from_level Its level.
    void interpolate_between_levels(vector_type &to_v,
                                    const int to_level,
                                    const vector_type &from_v,
                                    const int from_level);

    /// @brief Tests that density, Entropy, Pressure are all physical. Exits the
    /// program if not.
    /// @tparam v_type The vector type.
    /// @tparam dim The spatial dimension.
    /// @param u The vector to test.
    /// @param level the level at which the vector lives.
    /// @param where A string that tells where in the code this test is being
    /// done.
    void test_physicality(const vector_type u,
                          const int level,
                          std::string where = "");

    /// @brief Prints vector v at time t.
    /// @param v The vector to print.
    /// @param t The time.
    /// @param level The level at which this vector lives.
    /// @param fname Filenams to print.
    /// @param time_in_fname Do we include the time in the fname.
    /// @param cycle The Multigrid Cycle in which the vector is in.
    void print_solution(StateVector &v,
                        const double t = 0,
                        const int level = 0,
                        const std::string fname = "./test-output",
                        const unsigned int cycle = 0);

    /// @brief Returns the number of locally owned dofs at the specified level.
    /// @param level Level we are querying.
    /// @return Number of dofs owned on this process, at this level.
    unsigned int n_locally_owned_at_level(const int level) const;

  private:
    /// Creates all objects ryujin needs to run.
    void create_mg_levels();
    /// Calls prepare on all objects ryujin needs to run.
    void prepare_mg_objects();

  public: // Braid Required Routines
          /** @brief Apply the time stepping routine to the input vector @a u
               corresponding to time @a tstart, and return in the same vector @a u the
               computed result for time @a tstop. The values of @a tstart and @a tstop
               can be obtained from @a pstatus.
      
               @param[in,out] u Input: approximate solution at time @a tstart.
                                 Output: computed solution at time @a tstop.
               @param[in] ustop Previous approximate solution at @a tstop?
               @param[in] fstop Additional source at time @a tstop. May be set to
             NULL,       indicating no additional source.
          */
    braid_Int Step(braid_Vector u,
                   braid_Vector ustop,
                   braid_Vector fstop,
                   BraidStepStatus &pstatus) override;

    /** @brief Compute the residual at time @a tstop, given the approximate
        solutions at @a tstart and @a tstop. The values of @a tstart and @a
       tstop can be obtained from @a pstatus.

        @param[in]     u Input: approximate solution at time @a tstop.
        @param[in,out] r Input: approximate solution at time @a tstart.
                          Output: residual at time @a tstop.
    */
    braid_Int
    Residual(braid_Vector u, braid_Vector r, BraidStepStatus &pstatus) override;

    /// Allocate a new vector in @a *v_ptr, which is a deep copy of @a u.
    braid_Int Clone(braid_Vector u, braid_Vector *v_ptr) override;

    /** @brief Allocate a new vector in @a *u_ptr and initialize it with an
        initial guess appropriate for time @a t. If @a t is the starting time,
        this method should set @a *u_ptr to the initial value vector of the ODE
        problem. */
    braid_Int Init(braid_Real t, braid_Vector *u_ptr) override;


    /// De-allocate the vector @a u.
    braid_Int Free(braid_Vector u) override;

    /// Perform the operation: @a y = @a alpha * @a x + @a beta * @a y.
    braid_Int Sum(braid_Real alpha,
                  braid_Vector x,
                  braid_Real beta,
                  braid_Vector y) override;

    /// Compute in @a *norm_ptr an appropriate spatial norm of @a u.
    braid_Int SpatialNorm(braid_Vector u, braid_Real *norm_ptr) override;

    /// @see braid_PtFcnAccess.
    braid_Int Access(braid_Vector u, BraidAccessStatus &astatus) override;

    /// @see braid_PtFcnBufSize.
    braid_Int BufSize(braid_Int *size_ptr, BraidBufferStatus &bstatus) override;

    /// @see braid_PtFcnBufPack.
    braid_Int
    BufPack(braid_Vector u, void *buffer, BraidBufferStatus &bstatus) override;

    /// @see braid_PtFcnBufUnpack.
    braid_Int BufUnpack(void *buffer,
                        braid_Vector *u_ptr,
                        BraidBufferStatus &bstatus) override;

  public: // Public Data
    const MPI_Comm
        comm_x; // todo: can I make this const? uninitialized communicators now.
    std::vector<LevelType> levels; // instantiation
    std::vector<int> refinement_levels;
    std::vector<TimeLoopType> time_loops;
    braid_Int finest_level, coarsest_level;
    unsigned int n_fine_dofs;
    unsigned int n_locally_owned_dofs;
    bool print_solution_bool;
    braid_Int num_time;
    braid_Int cfactor; // FIXME: decide what types here? ryujin types or braid
                       // types? what about doubles??
    braid_Int max_iter;
    bool use_fmg;
    braid_Int n_relax;
    unsigned int n_cycles = 0;
    braid_Int access_level;

    // A vector used to store ALL levels of refinement offline_data for when we
    // need to interpolate vectors between levels.
    // TODO: refactor into a vector of paris, one the discretization, one the offline_data_vec
    std::vector<std::shared_ptr<DiscretizationType>> discretization_vec;
    std::vector<std::shared_ptr<OfflineDataType>> offline_data_vec;
    // std::vector<std::shared_ptr< todo make this pairs
    // A map that stores the index in the levels in the consistend
    std::map<int, int> level_map;

    std::string base_name;
    std::map<std::string, dealii::Timer> computing_timer;
    std::vector<std::vector<dealii::Tensor<1, dim>>> drag_history;
  };
}// Namespace mgrit