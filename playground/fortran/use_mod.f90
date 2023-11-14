! Compile the main program and link it with the module object file.
! The module file must be compiled first.

! gfortran -c print_mod.f90
! gfortran use_mod.f90 print_mod.o

program use_mod
  use print_mod
  implicit none

  real :: mat(10, 5)

  mat(:,:) = num_value

  call print_matrix(mat)

end program use_mod
