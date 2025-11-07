
#include "mgrit_description.h"
#include "my_app.template.h"

namespace mgrit
{

  /* instantiations of MGRIT Description*/
  template class MyApp<NUMBER, mgrit::Description, 1>;
  template class MyApp<NUMBER, mgrit::Description, 2>;
  template class MyApp<NUMBER, mgrit::Description, 3>;
  
} // namespace mgrit
