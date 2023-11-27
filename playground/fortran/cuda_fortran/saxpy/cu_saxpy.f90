module cu_saxpy
    use iso_c_binding
    implicit none

    ! Declare the Fortran interface for the C function
    interface
        subroutine cuda_saxpy(a, x, y, n) &
            bind(C, name="cuda_saxpy")
            use iso_c_binding
            real(c_float), value :: a
            real(c_float), dimension(*) :: x, y
            integer(c_int), value :: n
        end subroutine cuda_saxpy
    end interface

contains

    subroutine cusaxpy(a, x, y, n)
        implicit none

        real, intent(in) :: x(:), a
        real, intent(inout) :: y(:)
        integer, intent(in) :: n

        ! Declare the C variables

        real(c_float):: x_c(n), y_c(n), a_c
        integer(c_int) :: n_c

        ! Convert the Fortran variables to C variables

        x_c = x
        y_c = y
        n_c = n
        a_c = a

        ! Call the C function

        call cuda_saxpy(a_c, x_c, y_c, n_c)

        ! Convert the C array back to Fortran arrays

        y = y_c

    end subroutine cusaxpy
end module cu_saxpy
