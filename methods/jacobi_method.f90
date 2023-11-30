module jacobi_method
    use iso_c_binding
    implicit none

    real*8, parameter :: tolerance = 1.0d-5
    integer, parameter :: max_iter = 1000

    ! Declare the Fortran interfaces to C functions for the CUDA and OpenACC versions

    interface
        subroutine jacobi_cuda(A, b, x, size, iter, tol) &
            bind(C, name="jacobi_cuda")
            use iso_c_binding
            integer(c_int), value :: size, iter
            real(c_double), dimension(size,size) :: A
            real(c_double), dimension(size) :: b, x
            real(c_double), value :: tol
        end subroutine jacobi_cuda
    end interface

    interface
        subroutine jacobi_acc(A, b, x, size, iter, tol) &
            bind(C, name="jacobi_acc")
            use iso_c_binding
            integer(c_int), value :: size, iter
            real(c_double), dimension(size,size) :: A
            real(c_double), dimension(size) :: b, x
            real(c_double), value :: tol
        end subroutine jacobi_acc
    end interface

contains

    ! Sequential version of the Jacobi method
    subroutine jacobi(A, b, x, size_i, size_j)
        implicit none

        integer, intent(in) :: size_i, size_j
        double precision, intent(in) :: A(size_i,size_j), b(:)
        double precision, intent(inout) :: x(:)

        integer :: i, j, iter
        double precision :: error, new_x(size_i)

        do iter = 1, max_iter
            do i = 1, size_i
                new_x(i) = b(i)
                do j = 1, size_j
                    if (i /= j) then
                        new_x(i) = new_x(i) - A(i, j) * x(j)
                    end if
                end do
                if (A(i, i) /= 0.0d0) then
                    new_x(i) = new_x(i) / A(i, i)
                else
                    new_x(i) = 0.0d0
                end if
            end do

            ! calculate the error
            error = 0.0d0
            do i = 1, size_i
                error = error + dabs(new_x(i) - x(i))
            end do

            ! update the solution
            x = new_x

            if (error < tolerance) then
                exit
            end if

        end do


        if (iter == max_iter) then
            print *, "WARNING: Not converged after ", max_iter, " iterations"
        else
            print *, "Converged:", iter, " iterations"
        end if

        print *, "Error:        ", error

    end subroutine jacobi

    ! CUDA version of the Jacobi method

    subroutine cujacobi(A, b, x, size_i, size_j)
        implicit none

        integer, intent(in) :: size_i, size_j
        double precision, intent(in) :: A(size_i,size_j), b(:)
        double precision, intent(inout) :: x(:)

        call jacobi_wrapper(A, b, x, size_i, size_j, "cuda")

    end subroutine cujacobi

    ! OpenACC version of the Jacobi method

    subroutine accjacobi(A, b, x, size_i, size_j)
        implicit none

        integer, intent(in) :: size_i, size_j
        double precision, intent(in) :: A(size_i,size_j), b(:)
        double precision, intent(inout) :: x(:)

        call jacobi_wrapper(A, b, x, size_i, size_j, "acc")

    end subroutine accjacobi

    ! Wrapper subroutine to call the CUDA or OpenACC versions of the Jacobi method

    subroutine jacobi_wrapper(A, b, x, size_i, size_j, method)
        implicit none

        integer, intent(in) :: size_i, size_j
        double precision, intent(in) :: A(size_i,size_j), b(:)
        double precision, intent(inout) :: x(:)
        character(len=*), intent(in) :: method

        ! Declare the C variables
        integer(c_int) :: c_size, c_iter
        real(c_double), dimension(size_i, size_j), target :: c_A
        real(c_double), dimension(size_i), target :: c_b, c_x
        real(c_double) :: c_tolerance

        ! Convert the Fortran variables to C variables
        c_A = A
        c_b = b
        c_x = x
        c_size = size_i
        c_iter = max_iter
        c_tolerance = tolerance

        ! Call the C function
        if (method == "cuda") then
            call jacobi_cuda(c_A, c_b, c_x, c_size, c_iter, c_tolerance)
        else if (method == "acc") then
            call jacobi_acc(c_A, c_b, c_x, c_size, c_iter, c_tolerance)
        else
            print *, "ERROR: The '"//trim(method)//"' method is not implemented"
            stop
        end if

        ! Get the results back from the C function
        x = c_x

    end subroutine jacobi_wrapper

end module jacobi_method
