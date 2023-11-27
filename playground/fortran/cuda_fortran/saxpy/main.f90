! saxpy.f90

program main
    use cu_saxpy
    implicit none

    integer, parameter :: N = 40000
    real :: x(N), y(N), a

    x = 1.0; y = 2.0; a = 2.0

    call cusaxpy(a, x, y, N)

    write(*,*) 'Max error: ', maxval(abs(y-4.2))

end program main
