
#include "euler/description.h"
#include "euler_aeos/description.h"
#include "navier_stokes/description.h"
#include "scalar_conservation/description.h"
#include "shallow_water/description.h"


#include "my_app.template.h"

namespace mgrit
{
  /* instantiations Euler */
  template class MyApp<NUMBER, ryujin::Euler::Description, 1>;
  template class MyApp<NUMBER, ryujin::Euler::Description, 2>;
  template class MyApp<NUMBER, ryujin::Euler::Description, 3>;

//   /* instantiations EulerAEOS */
//   template class MyApp<NUMBER, ryujin::EulerAEOS::Description, 1>;
//   template class MyApp<NUMBER, ryujin::EulerAEOS::Description, 2>;
//   template class MyApp<NUMBER, ryujin::EulerAEOS::Description, 3>;

//   /* instantiations NavierStokes */
//   template class MyApp<NUMBER, ryujin::NavierStokes::Description, 1>;
//   template class MyApp<NUMBER, ryujin::NavierStokes::Description, 2>;
//   template class MyApp<NUMBER, ryujin::NavierStokes::Description, 3>;

//     /* instantiations ScalarConservation */
//   template class MyApp<NUMBER, ryujin::ScalarConservation::Description, 1>;
//   template class MyApp<NUMBER, ryujin::ScalarConservation::Description, 2>;
//   template class MyApp<NUMBER, ryujin::ScalarConservation::Description, 3>;

//     /* instantiations ShallowWater */
//   template class MyApp<NUMBER, ryujin::ShallowWater::Description, 1>;
//   template class MyApp<NUMBER, ryujin::ShallowWater::Description, 2>;
//   template class MyApp<NUMBER, ryujin::ShallowWater::Description, 3>;

} /* namespace ryujin */