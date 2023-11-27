subroutine test_saxpy
  use cuda_saxpy
  implicit none

  integer, parameter :: N = 40000
  real :: x(N), y(N), a

  x = 1.0; y = 2.0; a = 2.0

  call cuda_saxpy(x, y, a, N)
  
  write(*,*) 'Max error: ', maxval(abs(y-4.2))
end subroutine test_saxpy
